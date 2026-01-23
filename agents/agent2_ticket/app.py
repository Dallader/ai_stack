import os
import json
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
RAG_LIMIT = int(os.getenv("RAG_LIMIT", "3"))
MAX_CONTEXT_CHARS = int(os.getenv("OLLAMA_MAX_CONTEXT_CHARS", "4000"))
MAX_DOC_CHARS = int(os.getenv("OLLAMA_MAX_DOC_CHARS", "800"))

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_URL,
    num_ctx=OLLAMA_NUM_CTX,
    num_predict=OLLAMA_NUM_PREDICT,
    temperature=OLLAMA_TEMPERATURE
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
    
    query_embedding = embeddings.embed_query(query)
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
    
    return results

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

def generate_ticket_number() -> str:
    """Generate unique ticket number"""
    import time
    import random
    timestamp = int(time.time())
    random_num = random.randint(1000, 9999)
    return f"TICKET-{timestamp}-{random_num}"

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
    
    # Step 10.1: Analyze query with LLM
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
                "priority": classification["priority"],
                "priority_name": classification["priority_name"],
                "status": "created",
                "estimated_time": estimated_time,
                "created_at": ticket["created_at"]
            }
        )]
    )
    print(f"✓ Ticket {ticket_number} stored in Qdrant")
    
    # Step 10.6: Send notifications (logged for now)
    print(f"📧 Notifications sent for ticket {ticket_number}")
    
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
async def run(payload: dict):
    """Main endpoint following workflow steps from rules_and_workflow.json"""
    # Get step first to determine if input is required
    step = payload.get("step", "receive_query")
    
    # Steps that don't require an input query
    steps_without_input = ["greeting", "ask_for_help", "ask_details", "ask_create_ticket", 
                          "ask_continue", "farewell", "classify_and_create_ticket"]
    
    # Get query - allow None or empty string for steps that don't need it
    query = payload.get("input", "")
    original_query = payload.get("original_query", query)
    
    behavior = agent_behavior.get("behavior", {})
    
    # Debug logging
    print(f"📥 Received payload: {payload}")
    print(f"📥 Received: step='{step}' (type: {type(step)}), input='{query}', input_required={step not in steps_without_input}")
    
    # Validate input only for steps that require it
    if step not in steps_without_input and not query:
        raise HTTPException(status_code=400, detail=f"Input query is required for step '{step}'")
    
    # STEP 1: Greeting
    if step == "greeting":
        print("👋 Step 1: Greeting")
        greeting = behavior.get("greeting", "Witaj! Jestem Adam, asystentem WSB Merito.")
        return {
            "response": greeting,
            "step": "greeting",
            "next_step": "ask_for_help"
        }
    
    # STEP 2: Ask for help
    elif step == "ask_for_help":
        print("❓ Step 2: Ask for help")
        ask_help = behavior.get("ask_for_help", "W czym mogę Ci pomóc?")
        return {
            "response": ask_help,
            "step": "ask_for_help",
            "next_step": "receive_query"
        }
    
    # STEP 11: Ask if can continue helping
    elif step == "ask_continue":
        print("🔄 Step 11: Ask to continue")
        continue_conversation = payload.get("continue", False)
        
        if continue_conversation:
            ask_help = behavior.get("ask_for_help", "W czym mogę Ci pomóc?")
            return {
                "response": ask_help,
                "step": "ask_for_help",
                "next_step": "receive_query"
            }
        else:
            farewell = behavior.get("farewell", "Do zobaczenia!")
            return {
                "response": farewell,
                "step": "completed",
                "end": True
            }
    
    # STEP 9: Ask to create ticket (separate handler)
    elif step == "ask_create_ticket":
        print("🎫 Step 9: Ask to create ticket")
        create_ticket_request = payload.get("create_ticket", False)
        
        if create_ticket_request:
            # Move to ticket creation
            print("→ Step 10: User confirmed, creating ticket")
            print(behavior.get("classifying_ticket", "Klasyfikuję zgłoszenie..."))
            
            additional_info = payload.get("additional_info")
            input_query = payload.get("input", "")
            ticket = create_ticket(input_query or original_query, additional_info)
            
            # Format response using ticket_created template
            ticket_msg = behavior.get("ticket_created", "Zgłoszenie utworzone")
            response_text = ticket_msg.format(
                ticket_number=ticket["ticket_number"],
                priority=f"{ticket['priority']} ({ticket['priority_name']})",
                estimated_time=ticket["estimated_time"],
                status=ticket["status"]
            )
            
            # Ask if can help with something else
            ask_continue = behavior.get("ask_continue", "Czy mogę pomóc w czymś jeszcze?")
            
            return {
                "response": f"{response_text}\n\n{ask_continue}",
                "ticket": ticket,
                "step": "ticket_created",
                "next_step": "ask_continue",
                "documents_found": 0,
                "collection": COLLECTION
            }
        else:
            ask_continue = behavior.get("ask_continue", "Czy mogę pomóc w czymś jeszcze?")
            return {
                "response": ask_continue,
                "step": "ask_continue",
                "next_step": "receive_query"
            }
    
    # STEP 10: Classify and create ticket (alternative path)
    elif step == "classify_and_create_ticket":
        print("→ Step 10: Classifying and creating ticket")
        print(behavior.get("classifying_ticket", "Klasyfikuję zgłoszenie..."))
        
        additional_info = payload.get("additional_info")
        input_query = payload.get("input", "")
        ticket = create_ticket(input_query or original_query, additional_info)
        
        # Format response using ticket_created template
        ticket_msg = behavior.get("ticket_created", "Zgłoszenie utworzone")
        response_text = ticket_msg.format(
            ticket_number=ticket["ticket_number"],
            priority=f"{ticket['priority']} ({ticket['priority_name']})",
            estimated_time=ticket["estimated_time"],
            status=ticket["status"]
        )
        
        # Ask if can help with something else
        ask_continue = behavior.get("ask_continue", "Czy mogę pomóc w czymś jeszcze?")
        
        return {
            "response": f"{response_text}\n\n{ask_continue}",
            "ticket": ticket,
            "step": "ticket_created",
            "next_step": "ask_continue",
            "documents_found": 0,
            "collection": COLLECTION
        }
    
    # STEP 3: Receive query (already received in payload)
    # STEP 4: Search Qdrant database
    elif step in ["receive_query", "search_qdrant"]:
        print(f"🔎 Step 4: Searching Qdrant for: '{query}'")
        print(behavior.get("searching_message", "Searching..."))
        
        rag_results = search_rag_database(query, limit=RAG_LIMIT)
        
        # STEP 5: Generate answer with Ollama if documents found
        if rag_results:
            context = build_context(rag_results)
            
            print(f"✓ Step 5: Generating answer with {len(rag_results)} documents")
            
            # Get prompt template from JSON
            prompt_template = agent_behavior.get("llm_prompts", {}).get("generate_answer", "")
            prompt = prompt_template.format(context=context, query=query)
            
            response = llm.invoke(prompt)
            
            # STEP 11: Ask if can help with something else
            ask_continue = behavior.get("ask_continue", "Czy mogę pomóc w czymś jeszcze?")
            
            return {
                "response": f"{response.content}\n\n{ask_continue}",
                "rag_results": rag_results,
                "documents_found": len(rag_results),
                "step": "answer_provided",
                "next_step": "ask_continue",
                "collection": COLLECTION
            }
        
        # STEP 6: No results - ask for more details
        else:
            print("→ Step 6: No results, asking for details")
            ask_details = behavior.get("ask_for_details", "Czy możesz podać więcej szczegółów?")
            
            return {
                "response": ask_details,
                "step": "ask_details",
                "next_step": "receive_details",
                "original_query": query,
                "documents_found": 0,
                "collection": COLLECTION
            }
    
    # STEP 7: Receive additional details
    # STEP 8: Search with combined details
    elif step in ["receive_details", "search_with_details"]:
        combined_query = f"{original_query} {query}"
        print(f"🔎 Step 8: Searching with details: '{combined_query}'")
        print(behavior.get("searching_with_details", "Searching with additional info..."))
        
        rag_results = search_rag_database(combined_query, limit=RAG_LIMIT)
        
        # STEP 5: Generate answer if found
        if rag_results:
            context = build_context(rag_results)
            
            print(f"✓ Step 5: Generating answer with {len(rag_results)} documents")
            
            # Get prompt template from JSON
            prompt_template = agent_behavior.get("llm_prompts", {}).get("generate_answer_with_details", "")
            prompt = prompt_template.format(context=context, original_query=original_query, query=query)
            
            response = llm.invoke(prompt)
            ask_continue = behavior.get("ask_continue", "Czy mogę pomóc w czymś jeszcze?")
            
            return {
                "response": f"{response.content}\n\n{ask_continue}",
                "rag_results": rag_results,
                "documents_found": len(rag_results),
                "step": "answer_provided",
                "next_step": "ask_continue",
                "collection": COLLECTION
            }
        
        # STEP 9: Still no results - ask if create ticket
        else:
            print("→ Step 9: Still no results, asking to create ticket")
            ask_create = behavior.get("ask_create_ticket", "Czy chcesz utworzyć zgłoszenie?")
            
            return {
                "response": ask_create,
                "step": "ask_create_ticket",
                "next_step": "create_ticket",
                "original_query": original_query,
                "additional_info": query,
                "documents_found": 0,
                "collection": COLLECTION
            }
    
    # STEP 12: Farewell
    elif step == "farewell":
        farewell = behavior.get("farewell", "Do zobaczenia!")
        return {
            "response": farewell,
            "step": "completed",
            "end": True
        }
    
    else:
        return {
            "error": f"Unknown step: {step}",
            "documents_found": 0,
            "collection": COLLECTION
        }

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


