import os
import json
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from file_uploader import DocumentUploader, FileProcessor

app = FastAPI()
llm = ChatOllama(model="llama3", base_url="http://ollama:11434")
embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url="http://ollama:11434")

COLLECTION = os.getenv("COLLECTION", "agent2_tickets")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

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
rag_database = load_json_file("rag_database.json")
rules_workflow = load_json_file("rules_and_workflow.json")

# Initialize Qdrant collection
def init_qdrant_collection():
    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if COLLECTION not in collection_names:
            qdrant_client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=4096, distance=Distance.COSINE)
            )
            print(f"Created collection: {COLLECTION}")
        else:
            print(f"Collection {COLLECTION} already exists")
        
        # Always index RAG documents on startup to ensure fresh data
        index_rag_documents()
    except Exception as e:
        print(f"Error initializing Qdrant: {e}")

def index_rag_documents():
    """Index documents from rag_database.json into Qdrant"""
    try:
        print("📚 Starting document indexing...")
        documents = rag_database.get("documents", {})
        
        if not documents:
            print("⚠️ No documents found in rag_database.json!")
            return
        
        points = []
        point_id = 1
        
        for category, docs in documents.items():
            print(f"  📂 Category: {category} ({len(docs)} documents)")
            for doc in docs:
                text = f"{category}: {doc}"
                embedding = embeddings.embed_query(text)
                
                point = PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload={
                        "text": doc,
                        "category": category,
                        "type": "rag_document"
                    }
                )
                points.append(point)
                point_id += 1
        
        if points:
            qdrant_client.upsert(
                collection_name=COLLECTION,
                points=points
            )
            print(f"✅ Indexed {len(points)} documents into Qdrant collection '{COLLECTION}'")
        else:
            print("⚠️ No points to index!")
    except Exception as e:
        print(f"❌ Error indexing documents: {e}")

def search_rag_database(query: str, limit: int = 3) -> List[Dict]:
    """Search for relevant documents in Qdrant"""
    try:
        print(f"🔍 Searching Qdrant for query: '{query}'")
        query_embedding = embeddings.embed_query(query)
        print(f"📊 Generated embedding with {len(query_embedding)} dimensions")
        
        search_results = qdrant_client.search(
            collection_name=COLLECTION,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=0.3  # Obniżony próg dla lepszych wyników
        )
        
        print(f"✅ Found {len(search_results)} documents in Qdrant")
        
        results = []
        for result in search_results:
            doc = {
                "text": result.payload.get("text", ""),
                "category": result.payload.get("category", ""),
                "score": result.score
            }
            results.append(doc)
            print(f"  📄 Document: {doc['category']} - score: {doc['score']:.3f}")
        
        return results
    except Exception as e:
        print(f"❌ Error searching RAG database: {e}")
        return []

def assign_category_and_priority(ticket_text: str) -> Dict[str, str]:
    """Assign category and priority to ticket using LLM"""
    categories = categories_data.get("categories", [])
    categories_str = "\n".join([f"- {cat['name']} (Priorytet: {cat['priority']})" 
                                for cat in categories])
    
    prompt = f"""Przeanalizuj poniższe zgłoszenie i przypisz je do odpowiedniej kategorii.
    
Dostępne kategorie:
{categories_str}

Zgłoszenie: {ticket_text}

Odpowiedz w formacie JSON:
{{"category": "nazwa kategorii", "priority": "priorytet"}}
"""
    
    try:
        response = llm.invoke(prompt)
        # Try to parse JSON from response
        content = response.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        
        result = json.loads(content)
        return {
            "category": result.get("category", "Nieprzypisana"),
            "priority": result.get("priority", "Średni")
        }
    except Exception as e:
        print(f"Error assigning category: {e}")
        return {"category": "Nieprzypisana", "priority": "Średni"}

def create_ticket(query: str, additional_info: Optional[str] = None) -> Dict:
    """Create a ticket with category and priority"""
    full_text = f"{query}"
    if additional_info:
        full_text += f"\nDodatkowe informacje: {additional_info}"
    
    classification = assign_category_and_priority(full_text)
    
    ticket = {
        "query": query,
        "additional_info": additional_info,
        "category": classification["category"],
        "priority": classification["priority"],
        "status": "created"
    }
    
    # Store ticket in Qdrant
    try:
        embedding = embeddings.embed_query(full_text)
        qdrant_client.upsert(
            collection_name=COLLECTION,
            points=[PointStruct(
                id=hash(query) % (10**8),  # Simple ID generation
                vector=embedding,
                payload={
                    "type": "ticket",
                    "query": query,
                    "additional_info": additional_info,
                    "category": classification["category"],
                    "priority": classification["priority"]
                }
            )]
        )
    except Exception as e:
        print(f"Error storing ticket: {e}")
    
    return ticket

# Initialize collection on startup
@app.on_event("startup")
async def startup_event():
    print("🚀 Starting Agent2 Ticket System...")
    print(f"   Qdrant URL: {QDRANT_URL}")
    print(f"   Ollama URL: {OLLAMA_URL}")
    print(f"   Collection: {COLLECTION}")
    init_qdrant_collection()
    print("✅ Agent2 Ticket System ready!")

@app.get("/")
async def root():
    greeting = agent_behavior.get("behavior", {}).get("greeting", "Hello!")
    return {"message": greeting}

@app.get("/upload_form.html")
async def get_upload_form():
    """Serve the upload form HTML"""
    return FileResponse("upload_form.html")

@app.post("/run")
async def run(payload: dict):
    """Main endpoint for processing queries - ZAWSZE przeszukuje najpierw bazę Qdrant"""
    query = payload.get("input", "")
    step = payload.get("step", "initial")  # Track conversation step
    additional_info = payload.get("additional_info", None)
    
    if not query:
        raise HTTPException(status_code=400, detail="Input query is required")
    
    # KROK 1: ZAWSZE najpierw przeszukaj bazę danych Qdrant
    search_query = query
    original_query = query
    
    # Jeśli to kontynuacja, połącz zapytania
    if step == "need_details" and payload.get("original_query"):
        original_query = payload.get("original_query")
        search_query = f"{original_query} {query}"
    
    # Przeszukaj bazę dokumentów w Qdrant
    rag_results = search_rag_database(search_query, limit=5)
    
    # KROK 2: Jeśli znaleziono dokumenty, użyj LLM do wygenerowania odpowiedzi
    if rag_results:
        context = "\n".join([f"- {r['text']} (kategoria: {r['category']}, score: {r['score']:.2f})" 
                            for r in rag_results])
        
        # Przygotuj prompt z kontekstem
        if step == "need_details":
            prompt = f"""Na podstawie następujących dokumentów z bazy wiedzy odpowiedz na pytanie użytkownika.

Dokumenty z bazy wiedzy:
{context}

Pytanie początkowe: {original_query}
Dodatkowe informacje: {query}

Przeanalizuj dokumenty i udziel pomocnej, zwięzłej odpowiedzi po polsku. Jeśli dokumenty nie zawierają pełnej odpowiedzi, powiedz o tym."""
        else:
            prompt = f"""Na podstawie następujących dokumentów z bazy wiedzy odpowiedz na pytanie użytkownika.

Dokumenty z bazy wiedzy:
{context}

Pytanie: {query}

Przeanalizuj dokumenty i udziel pomocnej, zwięzłej odpowiedzi po polsku. Jeśli dokumenty nie zawierają pełnej odpowiedzi, powiedz o tym."""
        
        # Wywołaj LLM z kontekstem z Qdrant
        response = llm.invoke(prompt)
        
        return {
            "response": response.content,
            "rag_results": rag_results,
            "documents_found": len(rag_results),
            "step": "completed",
            "collection": COLLECTION
        }
    
    # KROK 3: Brak wyników - obsłuż według kroku
    if step == "initial":
        # Pierwsze zapytanie bez wyników - poproś o więcej szczegółów
        return {
            "response": "Nie znalazłem odpowiedzi w bazie dokumentów. Czy możesz podać więcej szczegółów lub przeformułować pytanie?",
            "step": "need_details",
            "original_query": query,
            "documents_found": 0,
            "collection": COLLECTION
        }
    
    elif step == "need_details":
        # Nadal brak wyników po dodatkowych szczegółach - utwórz zgłoszenie
        ticket = create_ticket(original_query, query)
        
        response_text = f"""Przepraszam, nie znalazłem odpowiedzi w bazie dokumentów nawet po dodatkowych szczegółach. 

Utworzyłem zgłoszenie:
- Kategoria: {ticket['category']}
- Priorytet: {ticket['priority']}

Powiadomiłem Ciebie i dział BOS e-mailem. Czy mogę pomóc w czymś jeszcze?"""
        
        return {
            "response": response_text,
            "ticket": ticket,
            "step": "ticket_created",
            "documents_found": 0,
            "collection": COLLECTION
        }
    
    # KROK 4: Bezpośrednie tworzenie zgłoszenia (tylko gdy użytkownik jawnie o to poprosi)
    elif step == "create_ticket":
        ticket = create_ticket(query, additional_info)
        return {
            "ticket": ticket,
            "response": f"Zgłoszenie utworzone. Kategoria: {ticket['category']}, Priorytet: {ticket['priority']}",
            "step": "ticket_created",
            "documents_found": 0,
            "collection": COLLECTION
        }
    
    return {
        "error": "Unknown step",
        "documents_found": 0,
        "collection": COLLECTION
    }

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    category: str = Form("Document")
):
    """Upload a file (PDF, DOCX, TXT, etc.) to Qdrant"""
    try:
        # Check if file type is supported
        if not file_processor.is_supported(file.filename):
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format. Supported: {', '.join(file_processor.supported_formats.keys())}"
            )
        
        # Read file content
        content = await file.read()
        
        # Upload to Qdrant
        result = document_uploader.upload_file(
            file_content=content,
            filename=file.filename,
            category=category
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload/batch")
async def upload_batch(files: List[UploadFile] = File(...), category: str = Form("Document")):
    """Upload multiple files at once"""
    results = []
    
    for file in files:
        try:
            if not file_processor.is_supported(file.filename):
                results.append({
                    "filename": file.filename,
                    "status": "skipped",
                    "reason": "unsupported format"
                })
                continue
            
            content = await file.read()
            result = document_uploader.upload_file(
                file_content=content,
                filename=file.filename,
                category=category
            )
            results.append(result)
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    return {"uploaded": len([r for r in results if r.get("status") == "success"]), "results": results}

@app.get("/stats")
async def get_stats():
    """Get collection statistics"""
    try:
        stats = document_uploader.get_collection_stats()
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    """Delete all chunks of a document by filename"""
    try:
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
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    try:
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
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "healthy", "collection": COLLECTION}

@app.post("/reindex")
async def reindex():
    """Force reindex all documents from rag_database.json"""
    try:
        print("🔄 Force reindexing documents...")
        index_rag_documents()
        
        # Get collection stats
        collection_info = qdrant_client.get_collection(COLLECTION)
        
        return {
            "status": "success",
            "message": "Documents reindexed",
            "collection": COLLECTION,
            "points_count": collection_info.points_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
