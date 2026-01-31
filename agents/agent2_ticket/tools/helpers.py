import os
from pathlib import Path
from uuid import uuid4
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple
import requests

# Document categories aligned with BOS taxonomy
CATEGORIES_PL: List[str] = [
    "Wnioski o przyjęcie na studia",
    "Zmiana danych osobowych",
    "Prośby o urlop lub zawieszenie",
    "Prośby o stypendia i świadczenia",
    "Dokumenty dotyczące zaliczeń i egzaminów",
    "Reklamacje i skargi",
    "Dokumenty finansowe",
    "Zaświadczenia i potwierdzenia",
    "Prośby o wsparcie administracyjne lub techniczne",
    "Pozostałe dokumenty",
]

try:
    import docx2txt
except Exception:
    docx2txt = None

try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=50):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap

        def split_documents(self, documents):
            chunks = []
            for doc in documents:
                text = getattr(doc, "page_content", str(doc)) or ""
                start = 0
                L = len(text)
                while start < L:
                    end = min(start + self.chunk_size, L)
                    chunk_text = text[start:end]
                    meta = dict(getattr(doc, "metadata", {}) or {})
                    chunks.append(type("Document", (), {"page_content": chunk_text, "metadata": meta})())
                    step = self.chunk_size - self.chunk_overlap
                    if step <= 0:
                        break
                    start += step
            return chunks

try:
    from langchain.embeddings import SentenceTransformerEmbeddings
except Exception:
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as _np

        class SentenceTransformerEmbeddings:
            def __init__(self, model_name: str = "all-MiniLM-L6-v2", model_kwargs: dict | None = None):
                model_kwargs = model_kwargs or {}
                self.model = SentenceTransformer(model_name, **model_kwargs)

            def embed_documents(self, texts):
                if isinstance(texts, list):
                    emb = self.model.encode(texts, show_progress_bar=False)
                    return [ _np.array(e).tolist() for e in emb]
                else:
                    emb = self.model.encode([texts], show_progress_bar=False)
                    return [ _np.array(emb[0]).tolist()]

            def embed_query(self, text):
                emb = self.model.encode([text], show_progress_bar=False)
                return _np.array(emb[0]).tolist()
    except Exception:
        SentenceTransformerEmbeddings = None

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as rest_models
except Exception:
    QdrantClient = None
    rest_models = None

def ollama_generate(prompt: str, model: str | None = None, ollama_url: str | None = None) -> str:
    # Try multiple base URLs: env, localhost, 127.0.0.1, ollama
    base_urls = []
    if ollama_url:
        base_urls.append(ollama_url.rstrip("/"))
    env_url = os.getenv("OLLAMA_URL")
    if env_url:
        base_urls.append(env_url.rstrip("/"))
    # Always try localhost and 127.0.0.1, then ollama (docker)
    base_urls += ["http://localhost:11434", "http://127.0.0.1:11434", "http://ollama:11434"]
    # Remove duplicates, preserve order
    seen = set()
    base_urls = [x for x in base_urls if not (x in seen or seen.add(x))]

    model = model or os.getenv("OLLAMA_MODEL", "llama3.1:latest")
    endpoints = [
        "/api/chat",
    ]

    errors = []
    for base in base_urls:
        for ep in endpoints:
            url = f"{base}{ep}"
            bodies = [
                {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": False},
            ]
            for body in bodies:
                try:
                    resp = requests.post(url, json=body, timeout=60)
                except Exception as e:
                    errors.append(f"{url} POST error: {e}")
                    continue
                if resp.status_code == 404:
                    errors.append(f"{url} 404")
                    continue
                try:
                    resp.raise_for_status()
                    data = resp.json()
                except Exception as e:
                    errors.append(f"{url} non-json or error: {e}")
                    continue

                if (
                    isinstance(data, dict)
                    and data.get("error")
                    and "model" in str(data.get("error")).lower()
                ):
                    model_list = []
                    model_eps = ["/api/models", "/v1/models", "/models"]
                    for mep in model_eps:
                        try:
                            murl = f"{base}{mep}"
                            mr = requests.get(murl, timeout=10)
                            if mr.status_code == 200:
                                md = mr.json()
                                if isinstance(md, list):
                                    for m in md:
                                        if isinstance(m, str):
                                            model_list.append(m)
                                        elif isinstance(m, dict) and m.get("name"):
                                            model_list.append(m.get("name"))
                                elif isinstance(md, dict) and md.get("models"):
                                    for m in md.get("models"):
                                        if isinstance(m, str):
                                            model_list.append(m)
                                        elif isinstance(m, dict) and m.get("name"):
                                            model_list.append(m.get("name"))
                        except Exception:
                            continue
                    if model_list:
                        new_model = model_list[0]
                        try:
                            return ollama_generate(prompt, model=new_model, ollama_url=base)
                        except Exception:
                            pass

                if isinstance(data, dict):
                    if "message" in data and isinstance(data["message"], dict):
                        return data["message"].get("content", "")
                    return str(data)

                if isinstance(data, list):
                    parts = []
                    for item in data:
                        if isinstance(item, dict):
                            parts.append(item.get("content") or item.get("text") or "")
                        elif isinstance(item, str):
                            parts.append(item)
                    return "".join(parts)

    return f"[Ollama generate error] No working endpoint. Tried bases: {base_urls}. Endpoints: {endpoints}. Errors: {'; '.join(errors[:5])}"


def load_documents_from_dir(dir_path: Path):
    docs = []
    try:
        if not dir_path.exists():
            return docs
    except PermissionError:
        try:
            import streamlit as _st

            _st.sidebar.warning(f"Permission denied accessing documents dir: {dir_path}")
        except Exception:
            pass
        return docs

    try:
        entries = sorted(dir_path.iterdir())
    except PermissionError:
        try:
            import streamlit as _st

            _st.sidebar.warning(f"Permission denied listing documents dir: {dir_path}")
        except Exception:
            pass
        return docs
    except Exception:
        return docs

    for p in entries:
        if p.is_dir():
            continue
        try:
            if p.suffix.lower() == ".pdf":
                try:
                    if PdfReader is None:
                        continue
                    reader = PdfReader(str(p))
                    text = "\n\n".join([page.extract_text() or "" for page in reader.pages])
                    docs.append(type("Document", (), {"page_content": text, "metadata": {"source": str(p)}})())
                except Exception:
                    continue
            elif p.suffix.lower() in {".txt", ".md"}:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        text = f.read()
                    docs.append(type("Document", (), {"page_content": text, "metadata": {"source": str(p)}})())
                except Exception:
                    continue
            elif p.suffix.lower() in {".docx", ".doc"}:
                try:
                    if docx2txt is None:
                        continue
                    text = docx2txt.process(str(p))
                    docs.append(type("Document", (), {"page_content": text, "metadata": {"source": str(p)}})())
                except Exception:
                    continue
            else:
                try:
                    with open(p, "r", encoding="utf-8", errors="ignore") as f:
                        text = f.read()
                    docs.append(type("Document", (), {"page_content": text, "metadata": {"source": str(p)}})())
                except Exception:
                    continue
        except Exception:
            continue
    return docs


def index_documents_to_qdrant(documents, collection_name: str | None = None, qdrant_url: str | None = None, model_name: str = "all-MiniLM-L6-v2"):
    if not documents:
        raise ValueError("No documents to index")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    if SentenceTransformerEmbeddings is None:
        raise RuntimeError("No embeddings implementation available")

    embeddings = SentenceTransformerEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})

    contents = [getattr(d, 'page_content', '') for d in texts]
    vectors = embeddings.embed_documents(contents)

    base_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    if QdrantClient is None:
        raise RuntimeError("qdrant-client is not available")

    client = QdrantClient(url=base_url)

    vector_size = len(vectors[0]) if vectors else 0

    # Top-level helpers

    def _infer_existing_size(info):
        existing_size = None
        if isinstance(info, dict):
            existing_size = (
                info.get("vectors_config", {}).get("size")
                or info.get("result", {}).get("vectors", {}).get("size")
            )
        else:
            existing_size = getattr(info, "vectors_config", None)
            if existing_size is not None:
                existing_size = getattr(existing_size, "size", None)
        try:
            existing_size = int(existing_size) if existing_size is not None else None
        except Exception:
            existing_size = None
        return existing_size

    def _ensure_collection(col_name: str, size: int):
        try:
            info = client.get_collection(col_name)
            existing_size = _infer_existing_size(info)
            if existing_size is None:
                # Unknown existing size — recreate to desired size
                if rest_models is not None:
                    client.recreate_collection(
                        collection_name=col_name,
                        vectors_config=rest_models.VectorParams(size=size, distance=rest_models.Distance.COSINE),
                    )
            elif existing_size != size:
                # Recreate to match expected vector size (destructive)
                if rest_models is not None:
                    client.recreate_collection(
                        collection_name=col_name,
                        vectors_config=rest_models.VectorParams(size=size, distance=rest_models.Distance.COSINE),
                    )
        except Exception:
            # Collection missing or client call failed — create it
            if rest_models is not None:
                try:
                    client.recreate_collection(
                        collection_name=col_name,
                        vectors_config=rest_models.VectorParams(size=size, distance=rest_models.Distance.COSINE),
                    )
                except Exception:
                    pass

    # Ensure both 'documents' and 'tickets' collections exist with correct vector size
    _ensure_collection("documents", vector_size)
    _ensure_collection("tickets", vector_size)

    points = []
    for i, vec in enumerate(vectors):
        payload = {"text": contents[i], "source": getattr(texts[i], 'metadata', {}).get('source')}
        if rest_models is not None:
            point = rest_models.PointStruct(id=uuid4().hex, vector=vec, payload=payload)
            points.append(point)

    if points:
        client.upsert(collection_name=collection_name, points=points)

    return {"client": client, "collection_name": collection_name, "embeddings": embeddings}


def chunk_text(text: str, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    chunks: List[str] = []
    if not text:
        return chunks
    step = max(1, chunk_size - chunk_overlap)
    for i in range(0, len(text), step):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def ensure_qdrant_collection(client: QdrantClient, name: str, vector_size: int) -> None:
    """Ensure the Qdrant collection exists with the requested vector size."""
    if client is None or rest_models is None:
        raise RuntimeError("qdrant-client is not available")

    try:
        info = client.get_collection(name)
        existing_size = getattr(getattr(info, "config", None), "params", None)
        existing_size = getattr(getattr(existing_size, "vectors", None), "size", None)
        try:
            existing_size = int(existing_size) if existing_size is not None else None
        except Exception:
            existing_size = None
        if existing_size != vector_size:
            client.recreate_collection(
                collection_name=name,
                vectors_config=rest_models.VectorParams(size=vector_size, distance=rest_models.Distance.COSINE),
            )
    except Exception:
        client.recreate_collection(
            collection_name=name,
            vectors_config=rest_models.VectorParams(size=vector_size, distance=rest_models.Distance.COSINE),
        )


def _extract_pdf_text(data: bytes) -> str:
    if PdfReader is None:
        return ""
    try:
        from io import BytesIO

        reader = PdfReader(BytesIO(data))
        return "\n\n".join([page.extract_text() or "" for page in reader.pages])
    except Exception:
        return ""


def extract_text_from_upload(filename: str, data: bytes) -> str:
    """Extract plain text from uploaded file bytes."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf_text(data)
    if suffix in {".txt", ".md"}:
        try:
            return data.decode("utf-8")
        except Exception:
            return data.decode("latin-1", errors="ignore")
    if suffix in {".doc", ".docx"}:
        if docx2txt is None:
            return ""
        try:
            from tempfile import NamedTemporaryFile

            with NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
                tmp.write(data)
                tmp.flush()
                return docx2txt.process(tmp.name)
        except Exception:
            return ""
    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def embed_chunks(
    model,
    chunks: List[str],
    *,
    is_query: bool = False,
    use_e5_prefix: bool = False,
    normalize: bool = True,
) -> List[List[float]]:
    """Encode text chunks using a SentenceTransformer-like model with optional e5 prefixes."""
    texts = chunks
    if use_e5_prefix:
        prefix = "query: " if is_query else "passage: "
        texts = [f"{prefix}{c}" for c in chunks]

    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=normalize)
    try:
        import numpy as _np

        return [ _np.array(vec).tolist() for vec in vectors ]
    except Exception:
        return [ list(vec) for vec in vectors ]


def search_similar_documents(
    client: QdrantClient,
    collection_name: str,
    query_vector: List[float],
    limit: int = 3,
    score_threshold: float = 0.82,
):
    """Search for similar documents in Qdrant with a score threshold."""
    if client is None:
        return []
    try:
        result = client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            score_threshold=score_threshold,
        ).points
        return result or []
    except Exception:
        return []


def categorize_text(text: str, categories: Optional[List[str]] = None, ollama_model: Optional[str] = None, ollama_url: Optional[str] = None) -> str:
    """Categorize text into one of the BOS categories using Ollama."""
    cats = categories or CATEGORIES_PL
    joined = "\n".join([f"- {c}" for c in cats])
    prompt = f"""Przypisz dokument do jednej z kategorii BOS. Wybierz dokładnie jedną kategorię z listy.

Kategorie:
{joined}

Tekst dokumentu (skrót):
{text[:1200]}

Zwróć tylko nazwę kategorii z listy, bez dodatkowych znaków."""
    response = ollama_generate(prompt, model=ollama_model, ollama_url=ollama_url)
    for cat in cats:
        if cat.lower() in response.lower():
            return cat
    return cats[-1]


def upsert_document_chunks(
    client: QdrantClient,
    collection_name: str,
    vectors: List[List[float]],
    chunks: List[str],
    base_payload: Dict[str, str],
):
    """Upsert chunked document vectors into Qdrant."""
    if client is None or rest_models is None:
        raise RuntimeError("qdrant-client is not available")

    points = []
    for idx, vector in enumerate(vectors):
        payload = dict(base_payload)
        payload.update({
            "chunk_id": idx,
            "text": chunks[idx],
        })
        point = rest_models.PointStruct(id=uuid4().hex, vector=vector, payload=payload)
        points.append(point)
    if points:
        client.upsert(collection_name=collection_name, points=points)
    return len(points)


def connect_vectorstore(collection_name: str | None = None, qdrant_url: str | None = None, embedding_dim: int | None = None):
    base_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    client = QdrantClient(url=base_url) if QdrantClient is not None else None
    try:
        if client is not None:
            client.get_collection(collection_name)
    except Exception:
        try:
            dim = embedding_dim or int(os.getenv("EMBEDDING_DIM", "0"))
        except Exception:
            dim = 0
        if dim > 0 and client is not None and rest_models is not None:
            # Ensure 'documents' and 'tickets' collections exist with requested dim
            def _ensure(col_name: str, size: int):
                try:
                    info = client.get_collection(col_name)
                    existing = None
                    if isinstance(info, dict):
                        existing = info.get("vectors_config", {}).get("size")
                    else:
                        existing = getattr(getattr(info, "vectors_config", None), "size", None)
                    try:
                        existing = int(existing) if existing is not None else None
                    except Exception:
                        existing = None
                    if existing is None or existing != size:
                        client.recreate_collection(
                            collection_name=col_name,
                            vectors_config=rest_models.VectorParams(size=size, distance=rest_models.Distance.COSINE),
                        )
                except Exception:
                    try:
                        client.recreate_collection(
                            collection_name=col_name,
                            vectors_config=rest_models.VectorParams(size=size, distance=rest_models.Distance.COSINE),
                        )
                    except Exception:
                        pass

            _ensure("documents", dim)
            _ensure("tickets", dim)
    return {"client": client, "collection_name": collection_name, "embeddings": None}


def create_ticket_in_qdrant(
    user_info: dict,
    chat_history: list,
    question: str,
    qdrant_url: str | None = None,
    model_name: str = "all-MiniLM-L6-v2"
) -> dict:
    """
    Create a ticket in Qdrant tickets collection with automatic categorization and priority.
    
    Args:
        user_info: Dict with keys: name, surname, email, index_number
        chat_history: List of chat messages [{"role": "user/assistant", "content": "..."}]
        question: The current question/issue
        qdrant_url: Qdrant URL (optional)
        model_name: Embedding model name
    
    Returns:
        Dict with ticket_id, category, priority, and status
    """
    from datetime import datetime
    
    if not user_info.get("name") or not user_info.get("email"):
        raise ValueError("User name and email are required")
    
    # Categorize ticket using AI
    categories = ["Technical Issue", "Academic Question", "Administrative", "Financial", "General Inquiry"]
    priorities = ["Low", "Medium", "High", "Critical"]
    
    # Build categorization prompt
    chat_summary = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in chat_history[-5:]])
    categorization_prompt = f"""Analyze this support ticket and provide:
1. Category (choose one: {', '.join(categories)})
2. Priority (choose one: {', '.join(priorities)})

Chat history:
{chat_summary}

Current question: {question}

Respond in format:
Category: <category>
Priority: <priority>
Reason: <brief reason>"""
    
    # Get categorization from Ollama
    try:
        ai_response = ollama_generate(categorization_prompt)
        
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
    except Exception:
        category = "General Inquiry"
        priority = "Medium"
    
    # Prepare ticket data
    ticket_id = uuid4().hex
    timestamp = datetime.utcnow().isoformat()
    
    # Create ticket embedding from question and chat history
    if SentenceTransformerEmbeddings is None:
        raise RuntimeError("No embeddings implementation available")
    
    embeddings = SentenceTransformerEmbeddings(model_name=model_name, model_kwargs={"device": "cpu"})
    
    # Combine question and recent chat for embedding
    ticket_text = f"{question}\n\nChat context: {chat_summary}"
    vector = embeddings.embed_query(ticket_text)
    
    # Connect to Qdrant
    base_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
    if QdrantClient is None:
        raise RuntimeError("qdrant-client is not available")
    
    client = QdrantClient(url=base_url)
    
    # Ensure tickets collection exists
    vector_size = len(vector)
    try:
        client.get_collection("tickets")
        # Verify vector size
        info = client.get_collection("tickets")
        existing_size = None
        if isinstance(info, dict):
            existing_size = info.get("vectors_config", {}).get("size")
        else:
            existing_size = getattr(getattr(info, "vectors_config", None), "size", None)
        try:
            existing_size = int(existing_size) if existing_size is not None else None
        except Exception:
            existing_size = None
        
        if existing_size and existing_size != vector_size and rest_models is not None:
            client.recreate_collection(
                collection_name="tickets",
                vectors_config=rest_models.VectorParams(size=vector_size, distance=rest_models.Distance.COSINE),
            )
    except Exception:
        if rest_models is not None:
            client.recreate_collection(
                collection_name="tickets",
                vectors_config=rest_models.VectorParams(size=vector_size, distance=rest_models.Distance.COSINE),
            )
    
    # Create ticket payload
    payload = {
        "ticket_id": ticket_id,
        "timestamp": timestamp,
        "status": "Open",
        "category": category,
        "priority": priority,
        "user_name": user_info.get("name", ""),
        "user_surname": user_info.get("surname", ""),
        "user_email": user_info.get("email", ""),
        "user_index_number": user_info.get("index_number", ""),
        "question": question,
        "chat_history": chat_history,
        "ticket_text": ticket_text
    }
    
    # Create point
    if rest_models is not None:
        point = rest_models.PointStruct(
            id=ticket_id,
            vector=vector,
            payload=payload
        )
        
        # Upload to Qdrant
        client.upsert(collection_name="tickets", points=[point])
    
    return {
        "ticket_id": ticket_id,
        "category": category,
        "priority": priority,
        "status": "Open",
        "timestamp": timestamp,
        "message": "Ticket created successfully"
    }
