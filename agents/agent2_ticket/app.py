import os
import json
import smtplib
from email.message import EmailMessage
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import ChatOllama, OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from file_uploader import DocumentUploader, FileProcessor

# Load host settings
def load_host_settings() -> dict:
    try:
        with open("host_settings.json", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load host_settings.json: {e}")
        return {}

host_settings = load_host_settings()
IS_LOCAL = host_settings.get("local ", True)  # Note: key has trailing space in JSON
VECTOR_SIZE = int(host_settings.get("vector_size", "4096"))
SCORE_THRESHOLD = float(host_settings.get("score_threshold", 0.3))
LOCAL_HOST = host_settings.get("local_host", "localhost")
SERVER_HOST = host_settings.get("server_host", "192.168.0.76")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "256"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))
OLLAMA_REQUEST_TIMEOUT = float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "45"))
RAG_LIMIT = int(os.getenv("RAG_LIMIT", "3"))
MAX_CONTEXT_CHARS = int(os.getenv("OLLAMA_MAX_CONTEXT_CHARS", "4000"))
MAX_DOC_CHARS = int(os.getenv("OLLAMA_MAX_DOC_CHARS", "800"))
FAST_RAG_ONLY = os.getenv("FAST_RAG_ONLY", "false").lower() in {"1", "true", "yes"}
MIN_RAG_SCORE = float(os.getenv("MIN_RAG_SCORE", "0.35"))
MIN_RAG_TOP_SCORE = float(os.getenv("MIN_RAG_TOP_SCORE", "0.45"))
MIN_RAG_SCORE_RATIO = float(os.getenv("MIN_RAG_SCORE_RATIO", "0.70"))

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL,
    num_ctx=OLLAMA_NUM_CTX,
    num_predict=OLLAMA_NUM_PREDICT,
    temperature=OLLAMA_TEMPERATURE,
    request_timeout=OLLAMA_REQUEST_TIMEOUT
)
embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url=OLLAMA_URL
)

COLLECTION = os.getenv("COLLECTION", "agent2_tickets")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL)

# Initialize file uploader
document_uploader = DocumentUploader(
    qdrant_url=QDRANT_URL,
    collection_name=COLLECTION,
    ollama_url=OLLAMA_URL,
    embeddings_model="nomic-embed-text"
)
file_processor = FileProcessor()

# Load JSON configuration files
def load_json_file(filename: str) -> dict:
    try:
        with open(f"json/{filename}", "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return {}

agent_behavior = load_json_file("agent_behavior.json")
categories_data = load_json_file("categories.json")
rules_workflow = load_json_file("rules_and_workflow.json")

# Initialize Qdrant collection
def init_qdrant_collection():
    collections = qdrant_client.get_collections().collections
    collection_names = [col.name for col in collections]
    
    if COLLECTION not in collection_names:
        qdrant_client.create_collection(
            collection_name=COLLECTION,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
        )
        print(f"Created collection: {COLLECTION} with {VECTOR_SIZE} dimensions")
    else:
        print(f"Collection {COLLECTION} already exists")

def preprocess_query(query: str) -> str:
    """Preprocess and expand query for better search results"""
    # Remove common filler words that don't add semantic value
    filler_words = ['proszę', 'chciałbym', 'czy możesz', 'jak mogę']
    processed = query.lower()
    for word in filler_words:
        processed = processed.replace(word, '')
    return processed.strip()

def search_rag_database(query: str, limit: int = RAG_LIMIT) -> List[Dict]:
    """Search for relevant documents in Qdrant with improved relevance"""
    print(f" Searching Qdrant for query: '{query}'")
    
    # Preprocess query for better matching
    processed_query = preprocess_query(query)
    print(f" Processed query: '{processed_query}'")
    
    query_embedding = embeddings.embed_query(processed_query)
    print(f" Generated embedding with {len(query_embedding)} dimensions")
    
    search_results = qdrant_client.query_points(
        collection_name=COLLECTION,
        query=query_embedding,
        limit=limit,
        score_threshold=SCORE_THRESHOLD  # Use threshold from host_settings
    ).points
    
    print(f" Found {len(search_results)} documents in Qdrant")
    
    results = []
    for result in search_results:
        doc_text = result.payload.get("text", "")
        if len(doc_text) > MAX_DOC_CHARS:
            doc_text = doc_text[:MAX_DOC_CHARS] + "..."
        doc = {
            "text": doc_text,
            "category": result.payload.get("category", ""),
            "score": result.score
        }
        results.append(doc)
        print(f" Document: {doc['category']} - score: {doc['score']:.3f}")

    # Hard relevance filtering to avoid mismatched answers
    if not results:
        return results

    top_score = max(r["score"] for r in results)
    if top_score < MIN_RAG_TOP_SCORE:
        print(f" Top score {top_score:.3f} below MIN_RAG_TOP_SCORE {MIN_RAG_TOP_SCORE:.3f} -> no results")
        return []

    min_allowed = max(MIN_RAG_SCORE, top_score * MIN_RAG_SCORE_RATIO)
    filtered = [r for r in results if r["score"] >= min_allowed]
    print(f" Relevance filter: top={top_score:.3f}, min_allowed={min_allowed:.3f}, kept={len(filtered)}/{len(results)}")
    return filtered

def assign_priority(ticket_text: str) -> Dict[str, str]:
    """Assign priority to ticket using LLM"""
    priorities = categories_data.get("priorities", [])
    priorities_str = "\n".join([f"- {p['level']} ({p['name']})" for p in priorities])
    
    # Get prompt template from JSON
    prompt_template = agent_behavior.get("llm_prompts", {}).get("priority_classification", "")
    prompt = prompt_template.format(priorities_str=priorities_str, ticket_text=ticket_text)
    
    response = llm.invoke(prompt)
    content = response.content.strip().upper()
    
    # Extract priority level (P1, P2, P3, P4)
    for p in priorities:
        if p['level'] in content:
            return {
                "priority": p['level'],
                "priority_name": p['name']
            }
    
    # Default to P3 if can't determine
    return {"priority": "P3", "priority_name": "Medium"}

def assign_category(ticket_text: str) -> Dict[str, str]:
    """Assign category to ticket using LLM with keyword fallback"""
    categories = categories_data.get("categories", [])

    def get_default_category() -> Dict[str, str]:
        for c in categories:
            if c.get("id") == "OTHER":
                return {"category": c.get("id", "OTHER"), "category_name": c.get("name", "Inne")}
        return {"category": "OTHER", "category_name": "Inne"}

    if not categories:
        return get_default_category()

    categories_str = "\n".join([
        f"- {c.get('id')}: {c.get('name')} – {c.get('description', '')}" for c in categories
    ])

    prompt_template = agent_behavior.get("llm_prompts", {}).get("category_classification", "")
    if prompt_template:
        prompt = prompt_template.format(categories_str=categories_str, ticket_text=ticket_text)
        try:
            response = llm.invoke(prompt)
            content = response.content.strip().upper()
            for c in categories:
                if c.get("id", "").upper() in content:
                    return {
                        "category": c.get("id", "OTHER"),
                        "category_name": c.get("name", "Inne")
                    }
        except Exception as e:
            print(f"⚠️ Category LLM error: {e}")

    # Keyword fallback
    text = ticket_text.lower()
    best_match = None
    best_score = 0
    for c in categories:
        keywords = c.get("keywords", [])
        score = sum(1 for k in keywords if k and k.lower() in text)
        if score > best_score:
            best_score = score
            best_match = c

    if best_match and best_score > 0:
        return {
            "category": best_match.get("id", "OTHER"),
            "category_name": best_match.get("name", "Inne")
        }

    return get_default_category()

def generate_ticket_number() -> str:
    """Generate unique ticket number"""
    import time
    import random
    timestamp = int(time.time())
    random_num = random.randint(1000, 9999)
    return f"TICKET-{timestamp}-{random_num}"

def send_ticket_email(ticket: Dict, to_email: str) -> Dict[str, str]:
    """Send ticket notification email if SMTP is configured."""
    smtp_host = os.getenv("SMTP_HOST", "").strip()
    smtp_port = int(os.getenv("SMTP_PORT", "587"))
    smtp_user = os.getenv("SMTP_USER", "").strip()
    smtp_password = os.getenv("SMTP_PASSWORD", "").strip()
    smtp_from = os.getenv("SMTP_FROM", smtp_user or "no-reply@localhost")
    smtp_tls = os.getenv("SMTP_TLS", "true").lower() in {"1", "true", "yes"}

    result = {
        "sent": False,
        "to": to_email
    }

    if not smtp_host:
        result["reason"] = "smtp_not_configured"
        print("⚠️ SMTP not configured. Skipping email send.")
        return result

    subject = f"[WSB Merito] Nowe zgłoszenie {ticket.get('ticket_number', '')}"
    body = (
        "Nowe zgłoszenie zostało utworzone.\n\n"
        f"Numer: {ticket.get('ticket_number', '')}\n"
        f"Kategoria: {ticket.get('category_name', '')} ({ticket.get('category', '')})\n"
        f"Priorytet: {ticket.get('priority', '')} ({ticket.get('priority_name', '')})\n"
        f"Status: {ticket.get('status', '')}\n"
        f"Szacowany czas: {ticket.get('estimated_time', '')}\n\n"
        f"Treść zgłoszenia:\n{ticket.get('query', '')}\n\n"
        f"Dodatkowe informacje:\n{ticket.get('additional_info') or 'brak'}\n"
    )

    msg = EmailMessage()
    msg["From"] = smtp_from
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
            server.ehlo()
            if smtp_tls:
                server.starttls()
                server.ehlo()
            if smtp_user and smtp_password:
                server.login(smtp_user, smtp_password)
            server.send_message(msg)
        result["sent"] = True
        print(f"📧 Ticket notification email sent to {to_email}")
    except Exception as e:
        result["reason"] = str(e)
        print(f"⚠️ Email send failed: {e}")

    return result

def build_email_status_line(email_info: Dict[str, str], fallback_email: str) -> str:
    if email_info.get("sent"):
        return f"📧 Wysłano zgłoszenie do: {email_info.get('to', fallback_email)}."
    reason = email_info.get("reason", "nieznany błąd")
    return f"⚠️ Nie udało się wysłać e-maila do {email_info.get('to', fallback_email)} ({reason})."

def get_estimated_time(priority: str) -> str:
    """Get estimated resolution time based on priority"""
    estimated_times = agent_behavior.get("ticket_estimated_times", {})
    return estimated_times.get(priority, "3-5 dni")

def create_ticket(query: str, additional_info: Optional[str] = None) -> Dict:
    """Create a ticket with priority, ticket number, and estimated resolution time"""
    import time
    
    full_text = f"{query}"
    if additional_info:
        full_text += f"\nDodatkowe informacje: {additional_info}"
    
    # Step 10.1: Classify category
    category_classification = assign_category(full_text)

    # Step 10.2: Assign priority
    classification = assign_priority(full_text)
    
    # Step 10.3: Generate ticket number
    ticket_number = generate_ticket_number()
    
    # Step 10.4: Estimate resolution time
    estimated_time = get_estimated_time(classification["priority"])
    
    ticket = {
        "ticket_number": ticket_number,
        "query": query,
        "additional_info": additional_info,
        "category": category_classification["category"],
        "category_name": category_classification["category_name"],
        "priority": classification["priority"],
        "priority_name": classification["priority_name"],
        "status": "created",
        "estimated_time": estimated_time,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Step 10.5: Store ticket in Qdrant
    embedding = embeddings.embed_query(full_text)
    qdrant_client.upsert(
        collection_name=COLLECTION,
        points=[PointStruct(
            id=hash(ticket_number),
            vector=embedding,
            payload={
                "type": "ticket",
                "ticket_number": ticket_number,
                "query": query,
                "additional_info": additional_info,
                "category": category_classification["category"],
                "category_name": category_classification["category_name"],
                "priority": classification["priority"],
                "priority_name": classification["priority_name"],
                "status": "created",
                "estimated_time": estimated_time,
                "created_at": ticket["created_at"]
            }
        )]
    )
    print(f"✓ Ticket {ticket_number} stored in Qdrant")
    
    # Step 10.6: Send notifications to BOS (best-effort)
    bos_email = os.getenv("BOS_EMAIL", "bos@merito.pl")
    email_info = send_ticket_email(ticket, bos_email)
    ticket["notification"] = email_info
    print(f"📧 Notification result for ticket {ticket_number}: {email_info}")
    
    return ticket

def build_context(rag_results: List[Dict]) -> str:
    """Build a bounded-size context string to avoid oversized prompts."""
    context = "\n\n".join([
        f"Dokument {i+1} (trafność: {r['score']:.2f}):\n{r['text']}\nKategoria: {r['category']}"
        for i, r in enumerate(rag_results)
    ])
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS] + "..."
    return context

def fast_answer_from_rag(rag_results: List[Dict]) -> str:
    """Return a quick response without calling the LLM."""
    top = rag_results[0]
    snippet = top["text"].strip()
    
    # Debug logging
    print(f"🔍 DEBUG fast_answer_from_rag:")
    print(f"   - Top result text length: {len(snippet)}")
    print(f"   - Top result text preview: {snippet[:200] if snippet else '[EMPTY]'}")
    print(f"   - Top result category: {top.get('category', 'N/A')}")
    print(f"   - Top result score: {top.get('score', 'N/A')}")
    
    if len(snippet) > 600:
        snippet = snippet[:600] + "..."
    return (
        "Znalazłem informacje w dokumentacji. Oto najtrafniejszy fragment:\n\n"
        f"{snippet}\n\n"
        "Jeśli potrzebujesz bardziej szczegółowej odpowiedzi, doprecyzuj pytanie."
    )

# Initialize collection on startup
@app.on_event("startup")
async def startup_event():
    print(" Starting Agent2 Ticket System...")
    print(f" Qdrant URL: {QDRANT_URL}")
    print(f" Ollama URL: {OLLAMA_URL}")
    print(f" Collection: {COLLECTION}")
    init_qdrant_collection()
    print(" Agent2 Ticket System ready!")

@app.get("/")
async def root():
    """Workflow Step 1 & 2: Greet user and ask for help"""
    behavior = agent_behavior.get("behavior", {})
    greeting = behavior.get("greeting", "Hello!")
    ask_for_help = behavior.get("ask_for_help", "How can I help?")
    
    return {
        "message": f"{greeting}\n{ask_for_help}",
        "step": "greeting",
        "next_step": "receive_query"
    }

@app.get("/upload_form.html")
async def get_upload_form():
    """Serve the upload form HTML"""
    return FileResponse("upload_form.html")

@app.get("/host_settings")
async def get_host_settings():
    """Get host settings for frontend"""
    return {
        "is_local": IS_LOCAL,
        "local_host": LOCAL_HOST,
        "server_host": SERVER_HOST,
        "vector_size": VECTOR_SIZE,
        "score_threshold": SCORE_THRESHOLD
    }

@app.post("/run")
async def run_agent(request: dict):
    """Main endpoint for agent operations"""
    try:
        step = request.get("step", "")
        user_input = request.get("input", "")
        
        if step == "greeting":
            return {
                "response": agent_behavior.get("responses", {}).get("greeting", "Dzień dobry! Jestem Twoim asystentem."),
                "step": "greeting"
            }
        
        elif step == "ask_for_help":
            return {
                "response": agent_behavior.get("responses", {}).get("ask_for_help", "W czym mogę Ci pomóc?"),
                "step": "ask_for_help"
            }
        
        elif step == "receive_query":
            rag_results = search_rag_database(user_input)
            
            if not rag_results:
                return {
                    "response": agent_behavior.get("responses", {}).get("no_results", "Nie znalazłem informacji. Czy możesz podać więcej szczegółów?"),
                    "step": "ask_details",
                    "documents_found": 0,
                    "original_query": user_input
                }
            
            # ... rest of receive_query logic
        
        elif step == "ask_details":
            return {
                "response": agent_behavior.get("responses", {}).get("ask_details", "Czy możesz podać więcej szczegółów dotyczących Twojego zapytania?"),
                "step": "ask_details",
                "original_query": request.get("original_query", "")
            }
        
        elif step == "receive_details":
            original_query = request.get("original_query", "")
            combined_query = f"{original_query} {user_input}"
            rag_results = search_rag_database(combined_query)
            
            if not rag_results:
                return {
                    "response": agent_behavior.get("responses", {}).get("still_no_results", "Niestety nadal nie mogę znaleźć odpowiedzi. Czy chcesz utworzyć zgłoszenie do BOS?"),
                    "step": "ask_create_ticket",
                    "documents_found": 0,
                    "original_query": original_query,
                    "additional_info": user_input
                }
            
            # ... rest of receive_details logic
        
        elif step == "search_with_details":
            # Similar to receive_details
            original_query = request.get("original_query", "")
            combined_query = f"{original_query} {user_input}"
            rag_results = search_rag_database(combined_query)
            
            return {
                "response": "Przetwarzam zapytanie z dodatkowymi szczegółami...",
                "rag_results": rag_results,
                "documents_found": len(rag_results),
                "step": "search_with_details"
            }
        
        # ... rest of your existing steps (farewell, ask_continue, etc.)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unknown step: {step}")
            
    except Exception as e:
        print(f"❌ Error in /run: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "step": request.get("step", "unknown")}
        )

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    category: str = Form("Document")
):
    """Upload a file (PDF, DOCX, TXT, etc.) to Qdrant"""
    print(f" Received upload request for: {file.filename}")
    print(f" Category: {category}")
    print(f" Collection: {COLLECTION}")
    print(f" Qdrant URL: {QDRANT_URL}")
    
    # Check if file type is supported
    if not file_processor.is_supported(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: {', '.join(file_processor.supported_formats.keys())}"
        )
    
    # Read file content
    content = await file.read()
    print(f" File size: {len(content)} bytes")
    
    # Upload to Qdrant
    result = document_uploader.upload_file(
        file_content=content,
        filename=file.filename,
        category=category
    )
    
    print(f" Upload result: {result}")
    
    if "error" in result:
        print(f" Upload error: {result['error']}")
        raise HTTPException(status_code=500, detail=result["error"])
    
    print(f" Upload successful: {result.get('chunks_uploaded', 0)} chunks uploaded")
    return result

@app.post("/upload/batch")
async def upload_batch(files: List[UploadFile] = File(...), category: str = Form("Document")):
    """Upload multiple files at once"""
    print(f" Batch upload request - {len(files)} files")
    print(f" Category: {category}")
    print(f" Collection: {COLLECTION}")
    
    results = []
    
    for file in files:
        print(f" Processing: {file.filename}")
        
        if not file_processor.is_supported(file.filename):
            results.append({
                "filename": file.filename,
                "status": "skipped",
                "reason": "unsupported format"
            })
            print(f" Skipped {file.filename} - unsupported format")
            continue
        
        content = await file.read()
        print(f" Read {len(content)} bytes from {file.filename}")
        
        result = document_uploader.upload_file(
            file_content=content,
            filename=file.filename,
            category=category
        )
        
        print(f"   Result for {file.filename}: {result}")
        results.append(result)
    
    successful = len([r for r in results if r.get("status") == "success"])
    print(f" Batch upload complete: {successful}/{len(files)} successful")
    
    return {"uploaded": successful, "results": results}

@app.get("/stats")
async def get_stats():
    """Get collection statistics"""
    stats = document_uploader.get_collection_stats()
    return stats

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete all chunks of a document by filename"""
    # Search for all points with this filename
    scroll_result = qdrant_client.scroll(
        collection_name=COLLECTION,
        scroll_filter={
            "must": [
                {
                    "key": "filename",
                    "match": {"value": filename}
                }
            ]
        },
        limit=1000
    )
    
    point_ids = [point.id for point in scroll_result[0]]
    
    if not point_ids:
        raise HTTPException(status_code=404, detail="Document not found")
    
    # Delete points
    qdrant_client.delete(
        collection_name=COLLECTION,
        points_selector=point_ids
    )
    
    return {
        "status": "success",
        "filename": filename,
        "chunks_deleted": len(point_ids)
    }

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    scroll_result = qdrant_client.scroll(
        collection_name=COLLECTION,
        limit=1000,
        with_payload=True
    )
    
    # Group by filename
    documents = {}
    for point in scroll_result[0]:
        payload = point.payload
        filename = payload.get("filename", "unknown")
        
        if filename not in documents:
            documents[filename] = {
                "filename": filename,
                "category": payload.get("category", "unknown"),
                "type": payload.get("type", "unknown"),
                "chunks": 0
            }
        
        documents[filename]["chunks"] += 1
    
    return {"documents": list(documents.values()), "total": len(documents)}

@app.post("/reindex")
async def reindex():
    """Reindex all documents from the documents folder"""
    import glob
    
    documents_dir = "documents"
    if not os.path.exists(documents_dir):
        return {
            "status": "error",
            "error": "documents folder not found"
        }
    
    # Get all supported files
    all_files = []
    for ext in ['.txt', '.pdf', '.docx', '.doc']:
        all_files.extend(glob.glob(f"{documents_dir}/**/*{ext}", recursive=True))
    
    if not all_files:
        return {
            "status": "warning",
            "message": "No documents found to reindex",
            "files_processed": 0
        }
    
    results = []
    for file_path in all_files:
        filename = os.path.basename(file_path)
        print(f"📄 Reindexing: {filename}")
        
        with open(file_path, 'rb') as f:
            content = f.read()
        
        result = document_uploader.upload_file(
            file_content=content,
            filename=filename,
            category="Document"
        )
        
        results.append(result)
        print(f"✓ {filename}: {result.get('chunks_uploaded', 0)} chunks")
    
    successful = len([r for r in results if r.get("status") == "success"])
    total_chunks = sum([r.get("chunks_uploaded", 0) for r in results if r.get("status") == "success"])
    
    return {
        "status": "success",
        "files_processed": len(all_files),
        "successful": successful,
        "total_chunks": total_chunks,
        "results": results
    }

@app.get("/health")
async def health():
    return {"status": "healthy", "collection": COLLECTION}
