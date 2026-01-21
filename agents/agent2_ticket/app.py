import os
import json
from typing import Optional, Dict, List
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from file_uploader import DocumentUploader, FileProcessor

app = FastAPI()
llm = ChatOllama(model="llama3", base_url="http://ollama:11434")
embeddings = OllamaEmbeddings(model="llama3", base_url="http://ollama:11434")

COLLECTION = os.getenv("COLLECTION", "agent2_tickets")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# Initialize Qdrant client
qdrant_client = QdrantClient(url=QDRANT_URL)

# Initialize file uploader
document_uploader = DocumentUploader(
    qdrant_url=QDRANT_URL,
    collection_name=COLLECTION,
    ollama_url=OLLAMA_URL
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
            
            # Index RAG documents
            index_rag_documents()
        else:
            print(f"Collection {COLLECTION} already exists")
    except Exception as e:
        print(f"Error initializing Qdrant: {e}")

def index_rag_documents():
    """Index documents from rag_database.json into Qdrant"""
    try:
        documents = rag_database.get("documents", {})
        points = []
        point_id = 1
        
        for category, docs in documents.items():
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
            print(f"Indexed {len(points)} documents into Qdrant")
    except Exception as e:
        print(f"Error indexing documents: {e}")

def search_rag_database(query: str, limit: int = 3) -> List[Dict]:
    """Search for relevant documents in Qdrant"""
    try:
        query_embedding = embeddings.embed_query(query)
        search_results = qdrant_client.search(
            collection_name=COLLECTION,
            query_vector=query_embedding,
            limit=limit,
            score_threshold=0.5
        )
        
        results = []
        for result in search_results:
            results.append({
                "text": result.payload.get("text", ""),
                "category": result.payload.get("category", ""),
                "score": result.score
            })
        return results
    except Exception as e:
        print(f"Error searching RAG database: {e}")
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
    init_qdrant_collection()

@app.get("/")
async def root():
    greeting = agent_behavior.get("behavior", {}).get("greeting", "Hello!")
    return {"message": greeting}

@app.post("/run")
async def run(payload: dict):
    """Main endpoint for processing queries"""
    query = payload.get("input", "")
    step = payload.get("step", "initial")  # Track conversation step
    additional_info = payload.get("additional_info", None)
    
    if not query:
        raise HTTPException(status_code=400, detail="Input query is required")
    
    # Step 1: Search RAG database
    if step == "initial":
        rag_results = search_rag_database(query)
        
        if rag_results:
            # Found relevant documents
            context = "\n".join([f"- {r['text']}" for r in rag_results])
            prompt = f"""Na podstawie następujących dokumentów odpowiedz na pytanie użytkownika.

Dokumenty:
{context}

Pytanie: {query}

Odpowiedz w sposób pomocny i zwięzły po polsku."""
            
            response = llm.invoke(prompt)
            return {
                "response": response.content,
                "rag_results": rag_results,
                "step": "completed",
                "collection": COLLECTION
            }
        else:
            # No results, ask for more details
            return {
                "response": "Nie znalazłem odpowiedzi w bazie dokumentów. Czy możesz podać dodatkowe szczegóły?",
                "step": "need_details",
                "original_query": query,
                "collection": COLLECTION
            }
    
    # Step 2: User provided additional info, search again
    elif step == "need_details":
        original_query = payload.get("original_query", query)
        combined_query = f"{original_query} {query}"
        
        rag_results = search_rag_database(combined_query)
        
        if rag_results:
            context = "\n".join([f"- {r['text']}" for r in rag_results])
            prompt = f"""Na podstawie następujących dokumentów odpowiedz na pytanie użytkownika.

Dokumenty:
{context}

Pytanie: {original_query}
Dodatkowe informacje: {query}

Odpowiedz w sposób pomocny i zwięzły po polsku."""
            
            response = llm.invoke(prompt)
            return {
                "response": response.content,
                "rag_results": rag_results,
                "step": "completed",
                "collection": COLLECTION
            }
        else:
            # Still no results, create ticket
            ticket = create_ticket(original_query, query)
            
            response_text = f"""Nie znalazłem odpowiedzi w bazie dokumentów. Utworzyłem zgłoszenie:

Kategoria: {ticket['category']}
Priorytet: {ticket['priority']}

Powiadomiłem Ciebie i dział BOS e-mailem. Czy mogę pomóc w czymś jeszcze?"""
            
            return {
                "response": response_text,
                "ticket": ticket,
                "step": "ticket_created",
                "collection": COLLECTION
            }
    
    # Direct ticket creation
    elif step == "create_ticket":
        ticket = create_ticket(query, additional_info)
        return {
            "ticket": ticket,
            "response": f"Zgłoszenie utworzone. Kategoria: {ticket['category']}, Priorytet: {ticket['priority']}",
            "step": "ticket_created",
            "collection": COLLECTION
        }
    
    return {"error": "Unknown step", "collection": COLLECTION}

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
