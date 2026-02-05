from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from pydantic import BaseModel
import re
import os
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
import numpy as np
from datetime import datetime
from uuid import uuid4

from tools.helpers import (
    CATEGORIES_PL,
    categorize_text,
    chunk_text,
    embed_chunks,
    ensure_qdrant_collection,
    extract_text_from_upload,
    search_similar_documents,
    upsert_document_chunks,
)

# Config
BASE_DIR = Path(__file__).parent
DOCS_DIR = BASE_DIR / "documents"
INCOMING_DIR = BASE_DIR / "incoming"
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
PROCESSED_DIR = BASE_DIR / "processed_documents"
LOGS_DIR = BASE_DIR / "logs"
SETTINGS_DIR = BASE_DIR / "settings"

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
TICKETS_COLLECTION = os.getenv("TICKETS_COLLECTION", "tickets")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_EMBED_URL = os.getenv("OLLAMA_EMBED_URL", OLLAMA_URL)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")

OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_EMBED_DIM = int(os.getenv("OLLAMA_EMBED_DIM", "768"))


def _parse_positive_int(value: Optional[str], default: int) -> int:
    try:
        parsed = int(value) if value is not None else default
        return default if parsed <= 0 else parsed
    except (TypeError, ValueError):
        return default


def _parse_timeout(value: Optional[str], default: Optional[int]) -> Optional[int]:
    try:
        if value is None:
            return default
        parsed = int(value)
        return None if parsed <= 0 else parsed
    except (TypeError, ValueError):
        return default


OLLAMA_REQUEST_TIMEOUT = _parse_timeout(os.getenv("OLLAMA_REQUEST_TIMEOUT"), 600)
_default_llm_timeout = None if OLLAMA_REQUEST_TIMEOUT is None else OLLAMA_REQUEST_TIMEOUT + 60
LLM_TIMEOUT_SECONDS = _parse_timeout(os.getenv("LLM_TIMEOUT_SECONDS"), _default_llm_timeout)
OLLAMA_NUM_PREDICT = _parse_positive_int(os.getenv("OLLAMA_NUM_PREDICT"), 768)
OLLAMA_NUM_CTX = _parse_positive_int(os.getenv("OLLAMA_NUM_CTX"), 4096)

LLM_OPTIONS = {
    "temperature": 0.2,
    "top_p": 0.9,
    "top_k": 25,
    "repeat_penalty": 1.1,
    "num_predict": OLLAMA_NUM_PREDICT,
    "num_ctx": OLLAMA_NUM_CTX,
    "num_thread": 10,
    "num_gpu": 1,
}

app = FastAPI(title="Agent WSB Merito API")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

VECTOR_SIZE = OLLAMA_EMBED_DIM
qdrant_client = QdrantClient(url=QDRANT_URL)

# Track the effective vector size returned by the embedding model
_effective_vector_size = VECTOR_SIZE

_documents_index_checked = False


def _load_prompt_config() -> Dict:
    cfg_path = SETTINGS_DIR / "prompt_config.json"
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {
            "system_instruction": (
                "You are BOS Assistant, the official virtual support assistant for university students. "
                "Be professional, calm, and supportive. Use only university documents; do not invent information."
            ),
            "critical_rules": [],
        }


PROMPT_CONFIG = _load_prompt_config()


def _ensure_documents_indexed_once():
    """Index known document folders on first access if the collection is empty."""
    global _documents_index_checked
    if _documents_index_checked:
        return
    _documents_index_checked = True
    try:
        info = qdrant_client.get_collection(QDRANT_COLLECTION)
        points_count = getattr(info, "points_count", None)
        try:
            points_count = int(points_count) if points_count is not None else None
        except Exception:
            points_count = None
    except Exception:
        points_count = None

    if not points_count:
        try:
            load_and_index_documents()
        except Exception as e:
            logging.warning("Auto-indexing documents failed: %s", e)


def _update_vector_size_from_embedding(vec: List[float]):
    """Align Qdrant collections to the embedding dimension returned by the model."""
    global VECTOR_SIZE, _effective_vector_size
    if not vec:
        return
    detected = len(vec)
    if detected != _effective_vector_size:
        logging.info("Detected embedding dimension %s (was %s); aligning collections.", detected, _effective_vector_size)
        _effective_vector_size = detected
        VECTOR_SIZE = detected
    try:
        ensure_collection_exists(QDRANT_COLLECTION, vector_size=_effective_vector_size)
    except Exception as e:
        logging.warning("Failed to ensure documents collection with size %s: %s", _effective_vector_size, e)
    try:
        ensure_tickets_collection_exists(vector_size=_effective_vector_size)
    except Exception as e:
        logging.warning("Failed to ensure tickets collection with size %s: %s", _effective_vector_size, e)


def ollama_embed(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using Ollama's /api/embeddings endpoint."""
    embeddings: List[List[float]] = []
    base_url = OLLAMA_EMBED_URL.rstrip("/")
    url = f"{base_url}/api/embeddings"
    for t in texts:
        payload = {"model": OLLAMA_EMBED_MODEL, "prompt": t}
        try:
            resp = requests.post(url, json=payload, timeout=OLLAMA_REQUEST_TIMEOUT)
            resp.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            status = http_err.response.status_code if http_err.response else None
            if status == 404:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "Endpoint /api/embeddings nie jest dostępny w usłudze Ollama. "
                        "Zaktualizuj obraz Ollama do najnowszej wersji i upewnij się, że model embeddingów jest pobrany."
                    ),
                )
            raise

        data = resp.json()
        vec = data.get("embedding")
        if not vec:
            raise HTTPException(status_code=500, detail="Brak embeddingu w odpowiedzi Ollama")
        embeddings.append(vec)
    if embeddings:
        expected_len = len(embeddings[0])
        for vec in embeddings:
            if len(vec) != expected_len:
                raise HTTPException(status_code=500, detail="Inconsistent embedding dimensions returned by model")
        _update_vector_size_from_embedding(embeddings[0])
    return embeddings


def encode_passages(texts: List[str]) -> List[List[float]]:
    return ollama_embed(texts)


def encode_query(text: str) -> List[float]:
    vec = ollama_embed([text])[0]
    _update_vector_size_from_embedding(vec)
    return vec


async def generate_with_timeout(prompt: str, context: str = "", timeout: Optional[int] = LLM_TIMEOUT_SECONDS) -> str:
    """Run ollama_generate with an optional async timeout and safe fallback message."""
    try:
        if timeout is None:
            return await asyncio.to_thread(ollama_generate, prompt, context)
        return await asyncio.wait_for(asyncio.to_thread(ollama_generate, prompt, context), timeout=timeout)
    except asyncio.TimeoutError:
        return (
            "Odpowiedź modelu zajmuje zbyt długo. Spróbuj ponownie za chwilę, podaj więcej szczegółów "
            "lub sprawdź dostępność usługi modelu."
        )

# In-memory session storage
sessions: Dict[str, Dict] = {}

class QueryRequest(BaseModel):
    question: str

class RunRequest(BaseModel):
    input: str
    session_id: Optional[str] = None

class TicketRequest(BaseModel):
    user_name: str
    user_surname: str
    user_email: str
    user_index_number: Optional[str] = ""
    question: str
    chat_history: List[Dict[str, str]] = []

class IndexRequest(BaseModel):
    pass

def ollama_generate(prompt: str, context: str = "") -> str:
    """Generate response using Ollama with fallback between /api/generate and /api/chat."""
    try:
        base_url = OLLAMA_URL.rstrip("/")

        system_instruction = PROMPT_CONFIG.get("system_instruction", "")
        critical_rules = PROMPT_CONFIG.get("critical_rules", []) or []

        rules_block = "\n".join([f"- {r}" for r in critical_rules])
        rules_text = f"\n\nKrytyczne zasady:\n{rules_block}" if rules_block else ""

        # Ground the prompt when context is present
        if context:
            formatted_prompt = (
                f"{system_instruction}{rules_text}\n\n"
                "Użyj poniższych dokumentów, aby odpowiedzieć jako BOS Assistant (pracownik BOS)."
                " Odpowiadaj tylko na podstawie treści dokumentów; jeśli czegoś brakuje, powiedz to wprost, zapytaj o doprecyzowanie,"
                " a dopiero potem ewentualnie zaproponuj zgłoszenie do BOS.\n\n"
                f"Dokumenty:\n{context}\n\n"
                f"Pytanie: {prompt}\n\n"
                "Odpowiedź: przygotuj logiczną, kompletną i czytelną odpowiedź (może być w punktach), w języku pytania"
                " (jeśli nie rozpoznasz, użyj polskiego). Dodaj krótkie cytaty/referencje z dokumentów, gdy pomagają."
            )
        else:
            formatted_prompt = (
                f"{system_instruction}{rules_text}\n\n"
                "Odpowiadaj jako BOS Assistant. Korzystaj wyłącznie z dostępnych informacji; jeśli czegoś brakuje, powiedz to i poproś o szczegóły.\n\n"
                f"Pytanie: {prompt}\n\n"
                "Odpowiedź: przygotuj logiczną, kompletną i czytelną odpowiedź (może być w punktach), w języku pytania"
                " (jeśli nie rozpoznasz, użyj polskiego)."
            )

        # First try the /api/generate endpoint
        gen_payload = {
            "model": OLLAMA_MODEL,
            "prompt": formatted_prompt,
            "stream": False,
            "options": dict(LLM_OPTIONS),
        }

        gen_url = f"{base_url}/api/generate"
        try:
            gen_resp = requests.post(gen_url, json=gen_payload, timeout=OLLAMA_REQUEST_TIMEOUT)
            if gen_resp.status_code == 404:
                raise RuntimeError("generate endpoint 404")
            gen_resp.raise_for_status()
            data = gen_resp.json()
            return data.get("response", "No response from model")
        except Exception:
            # Fallback to /api/chat (newer Ollama API)
            chat_url = f"{base_url}/api/chat"
            chat_payload = {
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "user", "content": formatted_prompt},
                ],
                "stream": False,
                "options": dict(LLM_OPTIONS),
            }
            chat_resp = requests.post(chat_url, json=chat_payload, timeout=OLLAMA_REQUEST_TIMEOUT)
            if chat_resp.status_code == 404:
                raise HTTPException(
                    status_code=502,
                    detail=(
                        "Ollama endpoint /api/chat not available at "
                        f"{chat_url}. Sprawdź OLLAMA_URL lub aktualizację obrazu Ollama."
                    ),
                )
            chat_resp.raise_for_status()
            data = chat_resp.json()
            # Ollama chat returns { message: { content: "..." } }
            if isinstance(data, dict) and isinstance(data.get("message"), dict):
                return data["message"].get("content", "No response from model")
            return data.get("response", "No response from model") if isinstance(data, dict) else str(data)

    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama at {OLLAMA_URL}. Please ensure Ollama service is running.",
        )
    except requests.exceptions.Timeout:
        wait_note = f"ponad {OLLAMA_REQUEST_TIMEOUT} sekund" if OLLAMA_REQUEST_TIMEOUT else "bardzo długo"
        return (
            f"Przepraszam, odpowiedź zajęła {wait_note}. "
            "Spróbuj zadać krótsze pytanie lub poczekaj, aż model się załaduje."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

def ensure_collection_exists(collection_name: str = QDRANT_COLLECTION, vector_size: Optional[int] = None):
    """Ensure Qdrant collection exists with correct configuration."""
    try:
        size = vector_size or VECTOR_SIZE
        ensure_qdrant_collection(qdrant_client, collection_name, size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant error: {str(e)}")


def _iter_source_files(base_dir: Path):
    """Yield (absolute_path, relative_path) for all files under base_dir."""
    if not base_dir.exists():
        return
    for file_path in base_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.name.startswith("."):
            continue
        try:
            rel_path = file_path.relative_to(base_dir)
        except Exception:
            rel_path = file_path.name
        yield file_path, rel_path


def load_and_index_documents() -> int:
    """Load documents from known folders and index to Qdrant."""
    sources = [
        (KNOWLEDGE_DIR, "knowledge"),
        (INCOMING_DIR, "incoming"),
        (DOCS_DIR, "documents"),  # legacy support
        (PROCESSED_DIR, "processed"),
    ]

    documents = []
    allowed_ext = {".pdf", ".txt", ".md", ".doc", ".docx", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".xls", ".xlsx"}

    for base_dir, origin in sources:
        for file_path, rel_path in _iter_source_files(base_dir):
            if file_path.suffix.lower() not in allowed_ext:
                continue
            try:
                data = file_path.read_bytes()
                text = extract_text_from_upload(file_path.name, data)
                if not text.strip():
                    continue
                documents.append({
                    "text": text,
                    "source": str(rel_path),
                    "path": str(file_path),
                    "origin": origin,
                })
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

    if not documents:
        raise HTTPException(status_code=404, detail="No documents found to index")

    ensure_collection_exists(QDRANT_COLLECTION, VECTOR_SIZE)

    total_points = 0
    for doc in documents:
        doc_id = uuid4().hex
        category = categorize_text(doc["text"], CATEGORIES_PL, ollama_model=OLLAMA_MODEL, ollama_url=OLLAMA_URL)
        suffix = Path(doc["path"]).suffix.lower()
        use_recursive = suffix in {".pdf", ".doc", ".docx"}
        chunk_size = 1000 if use_recursive else 800
        chunks = chunk_text(doc["text"], chunk_size=chunk_size, chunk_overlap=200, use_recursive=use_recursive)
        vectors = encode_passages(chunks)
        payload = {
            "doc_id": doc_id,
            "source": doc["source"],
            "path": doc["path"],
            "category": category,
            "origin": doc.get("origin", "documents"),
        }
        total_points += upsert_document_chunks(
            qdrant_client,
            QDRANT_COLLECTION,
            vectors,
            chunks,
            payload,
        )

    return total_points


def ingest_uploaded_document(filename: str, data: bytes, uploaded_by: Optional[str] = None) -> Dict[str, str]:
    """Handle student upload: save, compare with Qdrant, categorize, and index if new."""
    if not data:
        raise HTTPException(status_code=400, detail="Empty file upload")

    allowed_extensions = {".pdf", ".txt", ".md", ".docx", ".doc", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".xls", ".xlsx"}
    file_ext = Path(filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"File type {file_ext} not supported. Allowed: {', '.join(sorted(allowed_extensions))}",
        )

    INCOMING_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = INCOMING_DIR / filename
    raw_path.write_bytes(data)

    text = extract_text_from_upload(filename, data)
    if not text.strip():
        text = (
            f"Nie udało się odczytać treści z pliku {filename}. "
            "Jeśli to skan lub obraz, opisz krótko zawartość wiadomości, aby ułatwić odpowiedź."
        )

    ensure_collection_exists(QDRANT_COLLECTION, VECTOR_SIZE)

    query_vector = encode_query(text)
    matches = search_similar_documents(
        qdrant_client,
        QDRANT_COLLECTION,
        query_vector,
        limit=3,
        score_threshold=0.82,
    )

    if matches:
        best = matches[0]
        return {
            "status": "existing",
            "message": "Dokument już istnieje w bazie lub jest bardzo podobny.",
            "match_score": best.score,
            "match_source": best.payload.get("source"),
            "collection": QDRANT_COLLECTION,
        }

    category = categorize_text(text, CATEGORIES_PL, ollama_model=OLLAMA_MODEL, ollama_url=OLLAMA_URL)
    use_recursive = file_ext in {".pdf", ".doc", ".docx"}
    chunk_size = 1000 if use_recursive else 800
    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=200, use_recursive=use_recursive)
    if not chunks:
        chunks = [text]
    vectors = encode_passages(chunks)
    doc_id = uuid4().hex

    payload = {
        "doc_id": doc_id,
        "source": filename,
        "path": str(raw_path),
        "category": category,
        "origin": "student_upload",
        "uploaded_by": uploaded_by or "student",
    }

    points_added = upsert_document_chunks(
        qdrant_client,
        QDRANT_COLLECTION,
        vectors,
        chunks,
        payload,
    )

    processed_path = PROCESSED_DIR / f"{doc_id}_{Path(filename).stem}.txt"
    processed_path.write_text(text, encoding="utf-8", errors="ignore")

    return {
        "status": "indexed",
        "message": "Dokument został dodany do bazy.",
        "category": category,
        "doc_id": doc_id,
        "chunks_indexed": points_added,
        "collection": QDRANT_COLLECTION,
        "processed_copy": str(processed_path),
    }

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    for p in [DOCS_DIR, INCOMING_DIR, KNOWLEDGE_DIR, PROCESSED_DIR, LOGS_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    ensure_collection_exists()
    print(f"Connected to Qdrant at {QDRANT_URL}")
    print(f"Using collection: {QDRANT_COLLECTION}")
    print(f"Ollama URL: {OLLAMA_URL}")

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve the chat interface"""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/api")
async def api_info():
    return {
        "message": "Agent WSB Merito API is running",
        "qdrant_url": QDRANT_URL,
        "collection": QDRANT_COLLECTION,
        "ollama_url": OLLAMA_URL
    }


@app.get("/documents/categories")
async def list_categories():
    return {"categories": CATEGORIES_PL}


@app.post("/documents/upload")
async def upload_student_document(
    file: UploadFile = File(...),
    student_email: Optional[str] = Form(None),
    student_name: Optional[str] = Form(None),
):
    data = await file.read()
    result = ingest_uploaded_document(file.filename, data, uploaded_by=student_email or student_name)
    return result

@app.get("/collections")
async def list_collections():
    """List all Qdrant collections with document counts"""
    try:
        result = qdrant_client.get_collections()
        collections = []
        
        for c in result.collections:
            try:
                info = qdrant_client.get_collection(c.name)
                collections.append({
                    "name": c.name,
                    "points_count": info.points_count
                })
            except Exception as e:
                collections.append({
                    "name": c.name,
                    "points_count": None,
                    "error": str(e)
                })
        
        return {"collections": collections}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant error: {str(e)}")

class ClearHistoryRequest(BaseModel):
    session_id: str

@app.post("/clear_history")
async def clear_history(request: ClearHistoryRequest):
    """Clear chat history but keep user data"""
    try:
        if request.session_id in sessions:
            # Keep user data but clear chat history
            sessions[request.session_id]["chat_history"] = []
            return {
                "success": True,
                "message": "Chat history cleared",
                "user_data_preserved": sessions[request.session_id]["data_collection_complete"]
            }
        return {"success": False, "message": "Session not found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing history: {str(e)}")

@app.post("/run")
async def run_agent(request: RunRequest):
    """Main chat endpoint with session management and data collection"""
    try:
        user_input = request.input.strip()
        if not user_input:
            raise HTTPException(status_code=400, detail="Input cannot be empty")
        
        # Get or create session
        session_id = request.session_id or str(uuid4())
        if session_id not in sessions:
            sessions[session_id] = {
                "user_data": {
                    "name": None,
                    "surname": None,
                    "email": None,
                    "index_number": None
                },
                "data_collection_complete": True,
                "chat_history": [],
                "ticket_intent": False,
                "pending_issue": None,
            }

        # Backfill new fields for existing sessions
        session = sessions[session_id]
        session.setdefault("ticket_intent", False)
        session.setdefault("pending_issue", None)

        session = sessions[session_id]

        # Data collection complete - handle normal Q&A
        
        print(f"User input: '{user_input}'")
        
        # Check if user wants to create a ticket (BOS zgłoszenie)
        ticket_keywords = [
            "utwórz zgłoszenie", "utworzyć zgłoszenie", "zgłoszenie do bos", "stwórz zgłoszenie", "chcę zgłosić", "chciałbym zgłosić",
            "zrób zgłoszenie", "zrob zgłoszenie", "zrób ticket", "zrób zgłoszenie do bos"
        ]
        wants_ticket = any(keyword in user_input.lower() for keyword in ticket_keywords)

        def missing_fields(u):
            order = [
                ("imię", "name"),
                ("nazwisko", "surname"),
                ("email", "email"),
                ("numer indeksu", "index_number"),
            ]
            return [label for label, key in order if not u.get(key)]

        def _update_user_data_from_input(text: str, data: Dict[str, Optional[str]]):
            # Basic patterns to auto-fill when user writes everything in one message
            email_match = re.search(r"[\w.+'-]+@[\w.-]+\.[a-zA-Z]{2,}", text)
            if email_match and not data.get("email"):
                data["email"] = email_match.group(0)

            index_match = re.search(r"(?:indeks|index|nr indeksu|nr indexu)?\s*([0-9]{4,10})", text, re.IGNORECASE)
            if index_match and not data.get("index_number"):
                data["index_number"] = index_match.group(1)

            # Heuristic: if text has two words and looks like a name, fill missing slots
            if not data.get("name") or not data.get("surname"):
                tokens = [t for t in re.split(r"\s+", text.strip()) if t]
                if len(tokens) >= 2:
                    if not data.get("name"):
                        data["name"] = tokens[0].strip(",.")
                    if not data.get("surname"):
                        data["surname"] = tokens[1].strip(",.")

        def _has_issue_context(sess) -> bool:
            issue = sess.get("pending_issue") or ""
            if issue and len(issue.strip()) >= 24:
                return True
            # Check if there is any prior non-data user message to serve as issue description
            history = sess.get("chat_history", [])
            for msg in history:
                if msg.get("role") == "user" and not msg.get("is_data_collection"):
                    if len(msg.get("content", "").strip()) >= 24:
                        return True
            return False

        async def _try_rag_answer(issue_text: str):
            """Attempt to answer via RAG before creating a ticket."""
            if not issue_text or len(issue_text.strip()) < 12:
                return None, []
            try:
                query_vector = encode_query(issue_text)
                search_results = qdrant_client.query_points(
                    collection_name=QDRANT_COLLECTION,
                    query=query_vector,
                    limit=5,
                    score_threshold=0.2,
                ).points
                if not search_results:
                    return None, []

                context_parts = []
                sources: List[str] = []
                for result in search_results:
                    context_parts.append(f"[Źródło: {result.payload['source']}]\n{result.payload['text']}")
                    if result.payload['source'] not in sources:
                        sources.append(result.payload['source'])

                context = "\n\n---\n\n".join(context_parts)
                answer = await generate_with_timeout(issue_text, context=context)
                return answer, sources
            except Exception:
                return None, []

        if wants_ticket:
            session["ticket_intent"] = True
            if not session.get("pending_issue"):
                session["pending_issue"] = user_input

        if session.get("ticket_intent"):
            # Capture user's latest message into history for ticket context (regular so it counts as issue text)
            session["chat_history"].append({"role": "user", "content": user_input, "is_data_collection": False})

            # Try to auto-extract any details from free-form message
            _update_user_data_from_input(user_input, session["user_data"])

            # Build/accumulate issue text
            existing_issue = session.get("pending_issue") or ""
            issue_text = (existing_issue + "\n" + user_input).strip() if existing_issue else user_input
            session["pending_issue"] = issue_text

            # Try RAG answer before moving to ticket creation
            rag_answer, rag_sources = await _try_rag_answer(issue_text)
            if rag_answer:
                response = (
                    "Znalazłem odpowiedź w bazie wiedzy:\n\n"
                    f"{rag_answer}\n\n"
                    "Jeśli to nie rozwiązuje sprawy i chcesz zgłoszenie do BOS, napisz proszę ponownie, a zbierzemy dane."
                )
                session["ticket_intent"] = False
                session["data_collection_complete"] = True
                session["chat_history"].append({"role": "assistant", "content": response, "is_data_collection": False})
                return {
                    "response": response,
                    "session_id": session_id,
                    "data_collection_complete": True,
                    "clear_previous": False,
                    "sources": rag_sources,
                }

            # If no RAG answer and issue is too short, ask for more detail before collecting personal data
            if len(issue_text.strip()) < 48:
                response = (
                    "Nie znalazłem jeszcze odpowiedzi. Opisz proszę dokładniej problem (kontekst, terminy, system). "
                    "Po zebraniu opisu poproszę o dane do zgłoszenia."
                )
                session["data_collection_complete"] = False
                session["chat_history"].append({"role": "assistant", "content": response, "is_data_collection": True})
                return {
                    "response": response,
                    "session_id": session_id,
                    "data_collection_complete": False,
                    "clear_previous": False
                }

            missing = missing_fields(session["user_data"])

            if missing:
                prompts = {
                    "imię": "Podaj proszę swoje imię.",
                    "nazwisko": "Podaj proszę swoje nazwisko.",
                    "email": "Podaj proszę swój email studencki.",
                    "numer indeksu": "Podaj proszę numer indeksu (same cyfry).",
                }
                next_prompt = prompts.get(missing[0], "Podaj brakujące dane.")
                response = "Zbieram informacje z rozmowy. Aby utworzyć zgłoszenie do BOS, potrzebuję jeszcze: " + ", ".join(missing) + f"\n{next_prompt}"
                session["data_collection_complete"] = False
                session["chat_history"].append({"role": "assistant", "content": response, "is_data_collection": True})
                return {
                    "response": response,
                    "session_id": session_id,
                    "data_collection_complete": False,
                    "clear_previous": False
                }

            # All data present — proceed to create the ticket with status messages
            try:
                ensure_tickets_collection_exists()
                chat_history = session.get("chat_history", [])
                chat_summary = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in chat_history[-5:]])
                category, priority = categorize_ticket(issue_text, chat_history)
                ticket_id = uuid4().hex
                timestamp = datetime.now().isoformat()
                ticket_text = f"{issue_text}\n\nKontekst rozmowy: {chat_summary}"
                vector = encode_passages([ticket_text])[0]

                payload = {
                    "ticket_id": ticket_id,
                    "timestamp": timestamp,
                    "status": "Open",
                    "category": category,
                    "priority": priority,
                    "user_name": session["user_data"].get("name"),
                    "user_surname": session["user_data"].get("surname"),
                    "user_email": session["user_data"].get("email"),
                    "user_index_number": session["user_data"].get("index_number"),
                    "question": issue_text,
                    "chat_history": chat_history,
                    "ticket_text": ticket_text
                }

                point = rest_models.PointStruct(id=ticket_id, vector=vector, payload=payload)
                qdrant_client.upsert(collection_name=TICKETS_COLLECTION, points=[point])

                summary_lines = [
                    "Zbieram informacje z rozmowy...",
                    "Generuję zgłoszenie...",
                    "\nPodsumowanie zgłoszenia:",
                    f"- Numer: {ticket_id[:8]}",
                    f"- Kategoria: {category}",
                    f"- Priorytet: {priority}",
                    f"- Imię i nazwisko: {session['user_data'].get('name')} {session['user_data'].get('surname')}",
                    f"- Email: {session['user_data'].get('email')}",
                    f"- Nr indeksu: {session['user_data'].get('index_number')}",
                    f"- Opis: {issue_text}",
                    "Twoje zgłoszenie zostało przekazane do Biura Obsługi Studenta.",
                ]
                response = "\n".join(summary_lines)

                # Reset ticket intent state
                session["data_collection_complete"] = True
                session["ticket_intent"] = False
                session["pending_issue"] = None
                session["chat_history"].append({"role": "assistant", "content": response, "is_data_collection": True})

                return {
                    "response": response,
                    "session_id": session_id,
                    "data_collection_complete": True,
                    "clear_previous": False
                }
            except Exception as e:
                response = f"Przepraszam, wystąpił błąd podczas tworzenia zgłoszenia: {str(e)}"
                session["chat_history"].append({"role": "assistant", "content": response, "is_data_collection": True})
                return {
                    "response": response,
                    "session_id": session_id,
                    "data_collection_complete": False,
                    "clear_previous": False
                }

        # Normal Q&A flow
        _ensure_documents_indexed_once()

        query_vector = encode_query(user_input)

        search_results = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=5,
            score_threshold=0.2
        ).points

        response_header: Optional[str] = None

        if not search_results:
            status_lines = [
                " Szukam w bazie wiedzy...",
                " Nic nie znaleziono powyżej progu trafności",
                " Poproś o więcej szczegółów lub dodaj dokument."
            ]
            response = (
                "\n".join(status_lines) +
                "\n\nNie znalazłem wystarczających informacji. Podaj więcej szczegółów (kontekst, daty, kierunek), "
                "a jeśli chcesz, mogę później utworzyć zgłoszenie do BOS po zebraniu danych.")
        else:
            context_parts = []
            sources = []
            top_score = search_results[0].score if search_results else 0

            for result in search_results:
                context_parts.append(f"[Źródło: {result.payload['source']}]\n{result.payload['text']}")
                if result.payload['source'] not in sources:
                    sources.append(result.payload['source'])

            context = "\n\n---\n\n".join(context_parts)

            response_header = "Znalazłem poniższe informacje:"
            answer = await generate_with_timeout(user_input, context=context)
            response = answer
        
        # Store in chat history
        session["chat_history"].append({"role": "user", "content": user_input, "is_data_collection": False})
        session["chat_history"].append({"role": "assistant", "content": response, "is_data_collection": False})
        
        return {
            "response": response,
            "response_header": response_header,
            "session_id": session_id,
            "data_collection_complete": session.get("data_collection_complete", True),
            "clear_previous": False
        }
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Error in /run endpoint: {error_details}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session information"""
    if session_id in sessions:
        return {
            "session_id": session_id,
            "data_collection_complete": sessions[session_id]["data_collection_complete"],
            "user_data": sessions[session_id]["user_data"] if sessions[session_id]["data_collection_complete"] else {}
        }
    return {"session_id": None, "data_collection_complete": False, "user_data": {}}

@app.post("/query")
async def query_agent(request: QueryRequest):
    """Query the agent with RAG retrieval from Qdrant"""
    try:
        user_query = request.question.strip()
        if not user_query:
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        _ensure_documents_indexed_once()

        # Generate query embedding
        query_vector = encode_query(user_query)
        
        # Search Qdrant using query_points
        search_results = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=5,  # fetch more chunks to avoid missing specifics
            score_threshold=0.2  # lower threshold to capture narrower matches
        ).points
        
        if not search_results:
            return {
                "response": "I don't have any relevant information in my knowledge base to answer that question. Please try rephrasing or ask about topics covered in the uploaded documents.",
                "sources": [],
                "context_used": False
            }
        
        # Build context from retrieved chunks
        context_parts = []
        sources = []
        chunks_info = []  # For debugging
        
        for result in search_results:
            context_parts.append(f"[Source: {result.payload['source']}]\n{result.payload['text']}")
            chunks_info.append({
                "source": result.payload['source'],
                "score": result.score,
                "text_preview": result.payload['text'][:200] + "..."
            })
            if result.payload['source'] not in sources:
                sources.append(result.payload['source'])
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Log context for debugging
        print(f"\n=== QUERY: {user_query} ===")
        print(f"Found {len(search_results)} chunks")
        for info in chunks_info:
            print(f"  - {info['source']} (score: {info['score']:.3f}): {info['text_preview']}")
        
        # Use LLM to generate nice response from context
        response_text = await generate_with_timeout(user_query, context=context)
        
        return {
            "response": response_text,
            "sources": sources,
            "context_used": True,
            "chunks_retrieved": len(search_results),
            "chunks_info": chunks_info  # Include in response for debugging
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query error: {str(e)}")

@app.post("/index")
async def index_documents():
    """Index all documents from the documents directory"""
    try:
        ensure_collection_exists()
        chunks_indexed = load_and_index_documents()
        
        return {
            "message": f"Successfully indexed {chunks_indexed} chunks",
            "collection": QDRANT_COLLECTION
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing error: {str(e)}")

@app.post("/upload")
async def upload_document(file: UploadFile = File(...), student_email: Optional[str] = Form(None)):
    """Upload a document, check similarity, categorize, and index if new."""
    try:
        content = await file.read()
        return ingest_uploaded_document(file.filename, content, uploaded_by=student_email)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload error: {str(e)}")

@app.post("/upload-and-index")
async def upload_and_index_document(file: UploadFile = File(...)):
    """Upload a document and index it immediately (compat endpoint)."""
    try:
        content = await file.read()
        return ingest_uploaded_document(file.filename, content)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload and index error: {str(e)}")

def ensure_tickets_collection_exists(vector_size: Optional[int] = None):
    """Ensure tickets collection exists with correct vector size."""
    try:
        size = vector_size or VECTOR_SIZE
        ensure_qdrant_collection(qdrant_client, TICKETS_COLLECTION, size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant tickets collection error: {str(e)}")

def categorize_ticket(question: str, chat_history: List[Dict]) -> tuple:
    """Categorize ticket and assign priority using AI"""
    categories = ["Technical Issue", "Academic Question", "Administrative", "Financial", "General Inquiry"]
    priorities = ["Low", "Medium", "High", "Critical"]
    
    # Build categorization prompt
    chat_summary = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in chat_history[-5:]])
    
    categorization_prompt = f"""Analyze this support ticket and provide category and priority.

Chat history:
{chat_summary}

Current question: {question}

Choose category from: {', '.join(categories)}
Choose priority from: {', '.join(priorities)}

Consider:
- Technical issues: System errors, login problems, technical difficulties
- Academic: Course content, assignments, grades, academic policies
- Administrative: Enrollment, schedules, procedures
- Financial: Payments, fees, refunds
- General: Other inquiries

Priority guidelines:
- Critical: System down, urgent deadline issues
- High: Blocks important tasks, time-sensitive
- Medium: Important but not urgent
- Low: General questions, information requests

Respond in format:
Category: <category>
Priority: <priority>"""
    
    try:
        ai_response = ollama_generate(categorization_prompt, context="")
        
        # Parse AI response
        category = "General Inquiry"  # default
        priority = "Medium"  # default
        
        for line in ai_response.split("\n"):
            line = line.strip()
            if line.lower().startswith("category:"):
                for cat in categories:
                    if cat.lower() in line.lower():
                        category = cat
                        break
            elif line.lower().startswith("priority:"):
                for pri in priorities:
                    if pri.lower() in line.lower():
                        priority = pri
                        break
        
        return category, priority
    except Exception:
        return "General Inquiry", "Medium"

@app.post("/create-ticket")
async def create_ticket(request: TicketRequest):
    """Create a support ticket in Qdrant tickets collection"""
    try:
        # Validate required fields
        if not request.user_name or not request.user_email:
            raise HTTPException(status_code=400, detail="Name and email are required")
        
        # Ensure tickets collection exists
        ensure_tickets_collection_exists()
        
        # Categorize ticket
        category, priority = categorize_ticket(request.question, request.chat_history)
        
        # Generate ticket ID and timestamp
        ticket_id = uuid4().hex
        timestamp = datetime.now().isoformat()
        
        # Create embedding from question and chat history
        chat_summary = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" 
                                  for msg in request.chat_history[-5:]])
        ticket_text = f"{request.question}\n\nChat context: {chat_summary}"
        
        vector = encode_passages([ticket_text])[0]
        
        # Create ticket payload
        payload = {
            "ticket_id": ticket_id,
            "timestamp": timestamp,
            "status": "Open",
            "category": category,
            "priority": priority,
            "user_name": request.user_name,
            "user_surname": request.user_surname,
            "user_email": request.user_email,
            "user_index_number": request.user_index_number,
            "question": request.question,
            "chat_history": [{"role": msg.get("role", "user"), "content": msg.get("content", "")} 
                           for msg in request.chat_history],
            "ticket_text": ticket_text
        }
        
        # Create point and upload to Qdrant
        point = rest_models.PointStruct(
            id=ticket_id,
            vector=vector,
            payload=payload
        )
        
        qdrant_client.upsert(collection_name=TICKETS_COLLECTION, points=[point])
        
        return {
            "success": True,
            "ticket_id": ticket_id,
            "category": category,
            "priority": priority,
            "status": "Open",
            "timestamp": timestamp,
            "message": f"Ticket created successfully. Reference ID: {ticket_id[:8]}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ticket creation error: {str(e)}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Check Qdrant
        qdrant_client.get_collections()
        
        # Check Ollama
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        ollama_healthy = response.status_code == 200
        
        return {
            "status": "healthy",
            "qdrant": "connected",
            "ollama": "connected" if ollama_healthy else "disconnected"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
