import os
import glob
import json
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLLECTION = os.getenv("COLLECTION", "agent2_tickets")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "/app/documents")
# Default to a lightweight embedding model that Ollama hosts by default; override via env if desired.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
# Use a lightweight model for faster responses by default; override via env if needed.
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))
FORCE_RECREATE_COLLECTION = os.getenv("FORCE_RECREATE_COLLECTION", "false").lower() == "true"

llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,  # Maximum determinism for factual accuracy
    num_predict=150,  # Cap generation length to reduce latency
    top_k=5,  # Limit exploration for faster, focused decoding
    top_p=0.8,  # Slightly tighter nucleus sampling
)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
vector_store = None

prompt_config = {}
PROMPT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "settings", "prompt_config.json")
try:
    with open(PROMPT_CONFIG_PATH, "r", encoding="utf-8") as f:
        prompt_config = json.load(f)
        logger.info("Loaded prompt configuration from prompt_config.json")
except Exception as e:
    logger.warning(f"Could not load prompt config: {e}. Using defaults.")
    prompt_config = {
        "system_instruction": "Jesteś precyzyjnym asystentem dla studentów.",
        "critical_rules": [],
        "response_instruction": "ODPOWIEDŹ:",
        "student_deadline_phrases": ["ile mam"],
        "clarification_prefix": "PYTANIE STUDENTA:",
        "no_context_response": "Nie znalazłem informacji.",
        "no_relevant_docs_response": "Nie znalazłem informacji.",
        "error_response": "Przepraszam, wystąpił błąd."
    }


def load_documents_from_folder(folder_path: str):
    """Load all supported documents from a folder."""
    documents = []
    
    patterns = {
        "*.pdf": PyPDFLoader,
        "*.txt": TextLoader,
    }
    
    for pattern, loader_class in patterns.items():
        file_paths = glob.glob(os.path.join(folder_path, pattern))
        for file_path in file_paths:
            try:
                logger.info(f"Loading document: {file_path}")
                loader = loader_class(file_path)
                docs = loader.load()

                for doc in docs:
                    doc.metadata["source_file"] = os.path.basename(file_path)
                documents.extend(docs)
                logger.info(f"Successfully loaded {len(docs)} pages from {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
    
    return documents


def initialize_vector_store():
    """Initialize Qdrant vector store and load documents."""
    global vector_store
    
    try:
        if FORCE_RECREATE_COLLECTION:
            try:
                qdrant_client.delete_collection(COLLECTION)
                logger.info(f"Force deleted collection: {COLLECTION}")
            except Exception as e:
                logger.warning(f"Force delete collection failed or not present: {e}")

        collections = qdrant_client.get_collections().collections
        collection_exists = any(c.name == COLLECTION for c in collections)
        
        if not collection_exists:
            logger.info(f"Creating new collection: {COLLECTION}")
            qdrant_client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
            )
            
            documents = load_documents_from_folder(DOCUMENTS_PATH)
            
            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len,
                )
                chunks = text_splitter.split_documents(documents)
                logger.info(f"Split documents into {len(chunks)} chunks")
                
                vector_store = QdrantVectorStore.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    collection_name=COLLECTION,
                    url=f"http://{QDRANT_HOST}:{QDRANT_PORT}",
                    force_recreate=FORCE_RECREATE_COLLECTION,
                )
                logger.info(f"Successfully indexed {len(chunks)} document chunks")
            else:
                logger.warning("No documents found to load")
                vector_store = QdrantVectorStore(
                    client=qdrant_client,
                    collection_name=COLLECTION,
                    embedding=embeddings,
                )
        else:
            logger.info(f"Collection {COLLECTION} already exists, connecting...")
            vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=COLLECTION,
                embedding=embeddings,
            )
            
            collection_info = qdrant_client.get_collection(COLLECTION)
            logger.info(f"Collection has {collection_info.points_count} vectors")
            
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - initialize on startup."""
    logger.info("Starting Agent2 Ticket - initializing RAG system...")
    initialize_vector_store()
    logger.info("RAG system initialized successfully")
    yield
    logger.info("Shutting down Agent2 Ticket...")


app = FastAPI(lifespan=lifespan)

static_path = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_path), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def search_documents(query: str, k: int = 3):
    """Search for relevant documents in Qdrant."""
    if vector_store is None:
        return []
    
    try:
        results = vector_store.similarity_search_with_score(query, k=k)
        logger.info(f"Found {len(results)} results for query")
        return results
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return []


def generate_response(question: str, context_docs: list) -> str:
    """Generate a response using RAG - simple and effective."""
    
    if not context_docs:
        return prompt_config.get("no_context_response", "Nie znalazłem informacji na ten temat w dokumentach.")
    
    clarified_question = question
    student_phrases = prompt_config.get("student_deadline_phrases", ["ile mam"])
    if any(phrase in question.lower() for phrase in student_phrases):
        clarification_prefix = prompt_config.get("clarification_prefix", "PYTANIE STUDENTA:")
        clarified_question = f"{clarification_prefix} {question}"
        logger.info(f"Clarified question as student deadline query: {clarified_question}")
    
    for doc, score in context_docs:
        logger.info(f"Document score: {score}, source: {doc.metadata.get('source_file', 'unknown')}")
    
    relevant_docs = [(doc, score) for doc, score in context_docs if score < 0.8]
    
    if not relevant_docs:
        logger.info(f"No relevant documents found (best score: {context_docs[0][1]})")
        return prompt_config.get("no_relevant_docs_response", "Nie znalazłem wystarczająco trafnych informacji w dokumentach.")
    
    context_parts = []
    
    for doc, score in relevant_docs[:3]:
        logger.info(f"Using document with score: {score}")
        context_parts.append(doc.page_content)
    
    context = "\n\n".join(context_parts)

    system_instruction = prompt_config.get("system_instruction", "Jesteś asystentem.")
    critical_rules = prompt_config.get("critical_rules", [])
    response_instruction = prompt_config.get("response_instruction", "ODPOWIEDŹ:")
    
    rules_text = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(critical_rules)])
    
    prompt = f"""{system_instruction}

KONTEKST Z DOKUMENTÓW:
{context}

{clarified_question}

KRYTYCZNE ZASADY - CZYTAJ UWAŻNIE:
{rules_text}

{response_instruction}"""
    
    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()
        
        return answer
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return prompt_config.get("error_response", "Przepraszam, wystąpił błąd podczas generowania odpowiedzi.")


@app.post("/run")
async def run(payload: dict):
    """Main endpoint for processing student questions."""
    question = payload.get("input", "")
    
    if not question.strip():
        return {
            "response": "Proszę zadaj pytanie.",
            "collection": COLLECTION,
            "sources_found": 0
        }
    
    logger.info(f"Received question: {question}")
    
    # Handle meta-questions about documents
    doc_list_phrases = ["jakie masz dokumenty", "jakie dokumenty", "lista dokumentów", "pokaż dokumenty", "what documents"]
    if any(phrase in question.lower() for phrase in doc_list_phrases):
        documents = []
        for pattern in ["*.pdf", "*.txt"]:
            files = glob.glob(os.path.join(DOCUMENTS_PATH, pattern))
            documents.extend([os.path.basename(f) for f in files])
        
        doc_list = "\n".join([f"- {doc}" for doc in documents])
        return {
            "response": f"Mam dostęp do następujących dokumentów:\n{doc_list}",
            "collection": COLLECTION,
            "sources_found": len(documents)
        }
    
    context_docs = search_documents(question, k=5)
    
    response = generate_response(question, context_docs)
    
    return {
        "response": response,
        "collection": COLLECTION,
        "sources_found": len(context_docs)
    }


@app.post("/chat")
async def chat(payload: dict):
    """Chat endpoint - alias for /run for compatibility."""
    return await run(payload)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "collection": COLLECTION,
        "vector_store_ready": vector_store is not None
    }


@app.get("/", response_class=HTMLResponse)
async def get_chat_ui():
    """Serve the chat interface from external template."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "chat.html")
    with open(template_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    return html_content


@app.post("/reload-documents")
async def reload_documents():
    """Endpoint to reload documents from the documents folder."""
    try:
        qdrant_client.delete_collection(COLLECTION)
        logger.info(f"Deleted collection: {COLLECTION}")
        
        initialize_vector_store()
        
        return {
            "status": "success",
            "message": "Documents reloaded successfully",
            "collection": COLLECTION
        }
    except Exception as e:
        logger.error(f"Error reloading documents: {e}")
        return {
            "status": "error",
            "message": str(e)
        }


@app.get("/documents")
async def list_documents():
    """List all documents in the documents folder."""
    documents = []
    for pattern in ["*.pdf", "*.txt"]:
        files = glob.glob(os.path.join(DOCUMENTS_PATH, pattern))
        documents.extend([os.path.basename(f) for f in files])
    
    return {
        "documents": documents,
        "count": len(documents),
        "path": DOCUMENTS_PATH
    }