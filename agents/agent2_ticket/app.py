from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi import Request
from pydantic import BaseModel
import os
import asyncio
from pathlib import Path
from typing import List, Dict, Optional
import requests
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest_models
from sentence_transformers import SentenceTransformer
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
DOCS_DIR = BASE_DIR / "documents"  # legacy/support folder
INCOMING_DIR = BASE_DIR / "incoming"  # preferred for new uploads
KNOWLEDGE_DIR = BASE_DIR / "knowledge"
PROCESSED_DIR = BASE_DIR / "processed_documents"
LOGS_DIR = BASE_DIR / "logs"

QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "documents")
TICKETS_COLLECTION = os.getenv("TICKETS_COLLECTION", "tickets")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:latest")

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
USE_E5_PREFIX = "e5" in EMBEDDING_MODEL_NAME.lower()

app = FastAPI(title="Agent WSB Merito API")

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize embedding model globally
embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
VECTOR_SIZE = embedding_model.get_sentence_embedding_dimension()
qdrant_client = QdrantClient(url=QDRANT_URL)


def encode_passages(texts: List[str]) -> List[List[float]]:
    return embed_chunks(embedding_model, texts, use_e5_prefix=USE_E5_PREFIX, normalize=True)


def encode_query(text: str) -> List[float]:
    return embed_chunks(embedding_model, [text], is_query=True, use_e5_prefix=USE_E5_PREFIX, normalize=True)[0]

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

        # Ground the prompt when context is present
        if context:
            formatted_prompt = f"""Używając poniższych dokumentów, odpowiedz na pytanie studenta.

Dokumenty:
{context}

Pytanie: {prompt}

Odpowiedź: podaj zwięźle, w języku pytania (jeśli nie rozpoznasz, użyj polskiego)."""
        else:
            formatted_prompt = prompt

        # First try the /api/generate endpoint
        gen_payload = {
            "model": OLLAMA_MODEL,
            "prompt": formatted_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "top_p": 0.9,
                "top_k": 30,
                "repeat_penalty": 1.15,
                "num_predict": 256,
                "num_ctx": 8192,
                "num_thread": 8,
                "num_gpu": 1,
            },
        }

        gen_url = f"{base_url}/api/generate"
        try:
            gen_resp = requests.post(gen_url, json=gen_payload, timeout=120)
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
                "options": {
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "top_k": 30,
                    "repeat_penalty": 1.15,
                    "num_predict": 256,
                    "num_ctx": 8192,
                    "num_thread": 8,
                    "num_gpu": 1,
                },
            }
            chat_resp = requests.post(chat_url, json=chat_payload, timeout=120)
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
        return "Przepraszam, odpowiedź zajęła zbyt długo (ponad 120 sekund). Spróbuj zadać krótsze pytanie lub poczekaj, aż model się załaduje."
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")

def ensure_collection_exists(collection_name: str = QDRANT_COLLECTION, vector_size: int = VECTOR_SIZE):
    """Ensure Qdrant collection exists with correct configuration."""
    try:
        ensure_qdrant_collection(qdrant_client, collection_name, vector_size)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Qdrant error: {str(e)}")

def load_and_index_documents() -> int:
    """Load documents from known folders and index to Qdrant."""
    sources = [
        (KNOWLEDGE_DIR, "knowledge"),
        (INCOMING_DIR, "incoming"),
        (DOCS_DIR, "documents"),  # legacy support
        (PROCESSED_DIR, "processed"),
    ]

    documents = []
    allowed_ext = {".pdf", ".txt", ".md", ".doc", ".docx"}

    for base_dir, origin in sources:
        if not base_dir.exists():
            continue
        for file_path in base_dir.rglob("*"):
            if not file_path.is_file():
                continue
            if file_path.suffix.lower() not in allowed_ext:
                continue
            if file_path.name.startswith("."):
                continue
            try:
                data = file_path.read_bytes()
                text = extract_text_from_upload(file_path.name, data)
                if not text.strip():
                    continue
                documents.append({
                    "text": text,
                    "source": str(file_path.name),
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
        chunks = chunk_text(doc["text"], chunk_size=800, chunk_overlap=200)
        vectors = encode_passages(chunks)
        payload = {
            "doc_id": doc_id,
            "source": doc["source"],
            "path": doc["path"],
            "category": doc.get("origin", "documents"),
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

    allowed_extensions = {".pdf", ".txt", ".md", ".docx", ".doc"}
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
        raise HTTPException(status_code=400, detail="Nie udało się odczytać tekstu z pliku")

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
    chunks = chunk_text(text, chunk_size=800, chunk_overlap=200)
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
    asyncio.create_task(warmup_model())


async def warmup_model():
    """Load the Ollama model once at startup to reduce first-response latency."""
    try:
        # Give container a moment to ensure Ollama is reachable
        await asyncio.sleep(1)
        payload = {
            "model": OLLAMA_MODEL,
            "messages": [{"role": "user", "content": "ping"}],
            "stream": False,
            "options": {"num_predict": 50, "temperature": 0.2},
        }
        url = f"{OLLAMA_URL.rstrip('/')}/api/chat"
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        print("Ollama warmup complete")
    except Exception as e:
        print(f"Ollama warmup skipped: {e}")

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

@app.get("/student/{index_number}")
async def get_student_data(index_number: str):
    """Get student data by index number"""
    try:
        # Search for student by index number
        results = qdrant_client.scroll(
            collection_name="students",
            scroll_filter=rest_models.Filter(
                must=[
                    rest_models.FieldCondition(
                        key="index_number",
                        match=rest_models.MatchValue(value=index_number)
                    )
                ]
            ),
            limit=1
        )
        
        if not results[0]:
            raise HTTPException(status_code=404, detail=f"Student with index {index_number} not found")
        
        student = results[0][0].payload
        return {
            "student": student,
            "semester_averages": student.get("semester_averages", []),
            "year_averages": student.get("year_averages", []),
            "total_average": student.get("total_average", 0)
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching student: {str(e)}")

@app.post("/init-students")
async def initialize_students_db():
    """Initialize students database with sample data"""
    try:
        import subprocess
        result = subprocess.run(
            ["python3", "init_students_db.py"],
            cwd="/app",
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            return {
                "message": "Students database initialized successfully",
                "output": result.stdout
            }
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Initialization failed: {result.stderr}"
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

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
        
        # Check if asking about procedures/regulations (not personal grades)
        procedure_keywords = ["jak", "w jaki sposób", "procedura", "reklamacja", "zareklamować", "odwołać", "odwołanie", 
                             "złożyć", "składać", "wniosek", "regulamin", "zasady", "przepisy", "prawo"]
        asking_about_procedure = any(keyword in user_input.lower() for keyword in procedure_keywords)
        
        # Check if student is asking about their OWN grades/averages (not procedures)
        personal_grade_keywords = ["moja średnia", "moje oceny", "moja ocena", "jaką mam", "jaki mam", 
                                  "ile mam", "moje wyniki", "mój wynik", "mojej średniej", "moich ocen"]
        asking_about_personal_grades = any(keyword in user_input.lower() for keyword in personal_grade_keywords)
        
        # Only fetch student data if asking about personal grades AND not asking about procedures
        asking_about_grades = asking_about_personal_grades and not asking_about_procedure
        
        print(f"User input: '{user_input}'")
        print(f"Asking about procedure: {asking_about_procedure}")
        print(f"Asking about personal grades: {asking_about_personal_grades}")
        print(f"Will fetch student data: {asking_about_grades}")
        print(f"Index number: {session['user_data']['index_number']}")
        
        # Try to fetch student data if asking about grades
        student_data = None
        if asking_about_grades and session["user_data"]["index_number"]:
            try:
                print(f"Fetching student data for index: {session['user_data']['index_number']}")
                results = qdrant_client.scroll(
                    collection_name="students",
                    scroll_filter=rest_models.Filter(
                        must=[
                            rest_models.FieldCondition(
                                key="index_number",
                                match=rest_models.MatchValue(value=session["user_data"]["index_number"])
                            )
                        ]
                    ),
                    limit=1
                )
                if results[0]:
                    student_data = results[0][0].payload
                    print(f"Student data found: {student_data.get('name')} {student_data.get('surname')}")
                else:
                    print("No student data found in results")
            except Exception as e:
                print(f"Could not fetch student data: {e}")
                import traceback
                traceback.print_exc()
        
        # Check if user wants to create a ticket (BOS zgłoszenie)
        ticket_keywords = [
            "utwórz zgłoszenie", "utworzyć zgłoszenie", "zgłoszenie do bos", "stwórz zgłoszenie", "chcę zgłosić", "chciałbym zgłosić",
            "zrób zgłoszenie", "zrob zgłoszenie", "zrób ticket", "zrób zgłoszenie do bos"
        ]
        wants_ticket = any(keyword in user_input.lower() for keyword in ticket_keywords)

        def missing_fields(u):
            mapping = {"imię": "name", "nazwisko": "surname", "email": "email", "numer indeksu": "index_number"}
            return [k for k, v in mapping.items() if not u.get(v)]

        if wants_ticket:
            session["ticket_intent"] = True
            if not session.get("pending_issue"):
                session["pending_issue"] = user_input

        if session.get("ticket_intent"):
            # Try to extract any details from this message
            parse_user_details_from_text(user_input, session["user_data"])

            if user_details_complete(session["user_data"]):
                try:
                    ensure_tickets_collection_exists()
                    chat_history = session.get("chat_history", [])
                    chat_summary = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in chat_history[-5:]])
                    category, priority = categorize_ticket(user_input, chat_history)
                    ticket_id = uuid4().hex
                    timestamp = datetime.now().isoformat()
                    issue_text = session.get("pending_issue") or user_input
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

                    response = (
                        "✅ Zgłoszenie zostało utworzone!"\
                        f"\n\n📋 Numer zgłoszenia: {ticket_id[:8]}"\
                        f"\n📂 Kategoria: {category}"\
                        f"\n⚡ Priorytet: {priority}"\
                        f"\n📧 Email: {session['user_data'].get('email')}"\
                        f"\n🎓 Nr indeksu: {session['user_data'].get('index_number')}"\
                        "\nTwoje zgłoszenie zostało przekazane do Biura Obsługi Studenta."
                    )

                    # Reset ticket intent state
                    session["data_collection_complete"] = True
                    session["ticket_intent"] = False
                    session["pending_issue"] = None
                except Exception as e:
                    response = f"Przepraszam, wystąpił błąd podczas tworzenia zgłoszenia: {str(e)}"
            else:
                missing = missing_fields(session["user_data"])
                response = "Aby utworzyć zgłoszenie, podaj proszę: " + ", ".join(missing)
                session["data_collection_complete"] = False
                session["chat_history"].append({"role": "user", "content": user_input, "is_data_collection": True})
                session["chat_history"].append({"role": "assistant", "content": response, "is_data_collection": True})
                return {
                    "response": response,
                    "session_id": session_id,
                    "data_collection_complete": False,
                    "clear_previous": False
                }

        if student_data:
            # Student is asking about grades and we have their data
            # Build direct response with pre-calculated data (no LLM calculations)
            
            response_parts = [
                f"📊 Dane studenta: {student_data['name']} {student_data['surname']}",
                f"🎓 Program: {student_data['program']} - Rok {student_data['year']}",
                f"📋 Numer indeksu: {student_data['index_number']}",
                "",
                f"📈 **Średnia ogólna za wszystkie semestry: {student_data.get('total_average', 0)}**",
                ""
            ]
            
            # Check what specific info the user wants
            if "semestr" in user_input.lower():
                # Extract semester number if mentioned
                import re
                semester_match = re.search(r'semestr\w*\s+(\d+)', user_input.lower())
                if semester_match:
                    sem_num = int(semester_match.group(1))
                    # Find specific semester
                    sem_avg = next((s for s in student_data.get('semester_averages', []) if s['semester'] == sem_num), None)
                    if sem_avg:
                        response_parts.append(f"📚 **Średnia z semestru {sem_num}: {sem_avg['average']}**")
                        response_parts.append("")
                        # Show detailed grades for that semester
                        sem_data = next((s for s in student_data.get('semesters', []) if s['semester'] == sem_num), None)
                        if sem_data:
                            response_parts.append(f"Szczegółowe oceny - Semestr {sem_num} ({sem_data['year']}):")
                            for subject in sem_data['subjects']:
                                response_parts.append(f"  • {subject['name']}: {subject['grade']} ({subject['ects']} ECTS)")
                    else:
                        response_parts.append(f"Nie znaleziono danych dla semestru {sem_num}.")
                else:
                    # Show all semester averages
                    response_parts.append("📚 Średnie semestralne:")
                    for sem_avg in student_data.get('semester_averages', []):
                        response_parts.append(f"  • Semestr {sem_avg['semester']}: {sem_avg['average']}")
            
            if "rok" in user_input.lower() and "akademicki" in user_input.lower() or "roczn" in user_input.lower():
                response_parts.append("")
                response_parts.append("📅 Średnie roczne:")
                for year_avg in student_data.get('year_averages', []):
                    response_parts.append(f"  • Rok {year_avg['year']}: {year_avg['average']}")
            
            if "wszystkie" in user_input.lower() or "wszystkich" in user_input.lower():
                response_parts.append("")
                response_parts.append("📖 Wszystkie oceny:")
                for semester in student_data.get('semesters', []):
                    response_parts.append(f"\n**Semestr {semester['semester']} ({semester['year']})**")
                    sem_avg = next((s['average'] for s in student_data.get('semester_averages', []) if s['semester'] == semester['semester']), 0)
                    response_parts.append(f"Średnia semestralna: {sem_avg}")
                    for subject in semester['subjects']:
                        response_parts.append(f"  • {subject['name']}: {subject['grade']} ({subject['ects']} ECTS)")
            
            # If no specific query matched, show summary
            if len(response_parts) <= 6:
                response_parts.append("📚 Średnie semestralne:")
                for sem_avg in student_data.get('semester_averages', []):
                    response_parts.append(f"  • Semestr {sem_avg['semester']}: {sem_avg['average']}")
                response_parts.append("")
                response_parts.append("📅 Średnie roczne:")
                for year_avg in student_data.get('year_averages', []):
                    response_parts.append(f"  • Rok {year_avg['year']}: {year_avg['average']}")
            
            response = "\n".join(response_parts)
        else:
            # Normal Q&A flow
            # Generate query embedding
            query_vector = encode_query(user_input)
            
            # Search Qdrant using query_points
            search_results = qdrant_client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=query_vector,
                limit=3,  # Only top 3 most relevant chunks
                score_threshold=0.4  # Higher threshold for better quality
            ).points
            
            if not search_results:
                status_lines = [
                    "🔎 Szukam w bazie wiedzy...",
                    "⚠️ Nic nie znaleziono powyżej progu trafności",
                    "🙋 Poproś o więcej szczegółów lub dodaj dokument."
                ]
                response = (
                    "\n".join(status_lines) +
                    "\n\nNie znalazłem wystarczających informacji. Podaj więcej szczegółów (kontekst, daty, kierunek), "
                    "a jeśli chcesz, mogę później utworzyć zgłoszenie do BOS po zebraniu danych.")
            else:
                # Build context from retrieved chunks
                context_parts = []
                sources = []
                top_score = search_results[0].score if search_results else 0
                
                for result in search_results:
                    context_parts.append(f"[Źródło: {result.payload['source']}]\n{result.payload['text']}")
                    if result.payload['source'] not in sources:
                        sources.append(result.payload['source'])
                
                context = "\n\n---\n\n".join(context_parts)

                status_lines = [
                    "🔎 Szukam w bazie wiedzy...",
                    f"📚 Znalazłem {len(search_results)} fragmenty (top score: {top_score:.2f})",
                    "🤖 Generuję odpowiedź na podstawie znalezionych treści."
                ]
                
                # Use LLM to generate nice response from context
                answer = ollama_generate(user_input, context=context)
                response = "\n".join(status_lines) + "\n\n" + answer
        
        # Store in chat history
        session["chat_history"].append({"role": "user", "content": user_input, "is_data_collection": False})
        session["chat_history"].append({"role": "assistant", "content": response, "is_data_collection": False})
        
        return {
            "response": response,
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
        
        # Generate query embedding
        query_vector = encode_query(user_query)
        
        # Search Qdrant using query_points
        search_results = qdrant_client.query_points(
            collection_name=QDRANT_COLLECTION,
            query=query_vector,
            limit=3,  # Only top 3 most relevant chunks
            score_threshold=0.4  # Higher threshold for better quality
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
        response_text = ollama_generate(user_query, context=context)
        
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

def ensure_tickets_collection_exists():
    """Ensure tickets collection exists with correct vector size."""
    try:
        ensure_qdrant_collection(qdrant_client, TICKETS_COLLECTION, VECTOR_SIZE)
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

@app.get("/tickets")
async def list_tickets(status: Optional[str] = None, limit: int = 50):
    """List tickets from Qdrant"""
    try:
        ensure_tickets_collection_exists()
        
        # Scroll through tickets
        tickets = []
        offset = None
        
        while len(tickets) < limit:
            result = qdrant_client.scroll(
                collection_name=TICKETS_COLLECTION,
                limit=min(limit - len(tickets), 100),
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            
            points, next_offset = result
            
            if not points:
                break
            
            for point in points:
                payload = point.payload
                if status is None or payload.get("status") == status:
                    tickets.append({
                        "ticket_id": payload.get("ticket_id"),
                        "timestamp": payload.get("timestamp"),
                        "status": payload.get("status"),
                        "category": payload.get("category"),
                        "priority": payload.get("priority"),
                        "user_name": payload.get("user_name"),
                        "user_email": payload.get("user_email"),
                        "question": payload.get("question", "")[:100] + "..." if len(payload.get("question", "")) > 100 else payload.get("question", "")
                    })
            
            if next_offset is None:
                break
            offset = next_offset
        
        return {
            "tickets": tickets[:limit],
            "count": len(tickets)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching tickets: {str(e)}")

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