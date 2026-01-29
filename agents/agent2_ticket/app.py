import os
import glob
import json
import logging
import uuid
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from langchain_community.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
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
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:1b")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "512"))
FORCE_RECREATE_COLLECTION = os.getenv("FORCE_RECREATE_COLLECTION", "false").lower() == "true"
TICKET_STORE_PATH = os.getenv("TICKET_STORE_PATH", "/app/tickets.json")
AUTO_INDEX_ON_STARTUP = os.getenv("AUTO_INDEX_ON_STARTUP", "false").lower() == "true"

llm = ChatOllama(
    model=LLM_MODEL,
    base_url=OLLAMA_BASE_URL,
    temperature=0.0,  # Maximum determinism for factual accuracy
    num_predict=LLM_NUM_PREDICT,  # Allow longer answers while remaining bounded
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

# Load behavior config (assistant name / role)
behavior = {}
BEHAVIOR_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "settings", "behavior.json")
try:
    with open(BEHAVIOR_CONFIG_PATH, "r", encoding="utf-8") as f:
        behavior = json.load(f)
        logger.info("Loaded behavior configuration from behavior.json")
except Exception as e:
    logger.warning(f"Could not load behavior config: {e}. Using defaults.")
    behavior = {
        "assistant_name": "Asystent WSB Merito",
        "assistant_role": "helper dla studentów WSB Merito",
        "language": prompt_config.get("language", "pl")
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


def initialize_vector_store(auto_index: bool = False):
    """Initialize Qdrant vector store. If `auto_index` is True, load and index documents when collection is missing.

    By default (auto_index=False) the function will create/connect an empty collection and will NOT index
    documents. Use `/reload-documents` to explicitly trigger indexing.
    """
    global vector_store

    try:
        if FORCE_RECREATE_COLLECTION and auto_index:
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

            if auto_index:
                documents = load_documents_from_folder(DOCUMENTS_PATH)

                if documents:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=256,
                        chunk_overlap=50,
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
                logger.info("Auto-index on startup disabled; created empty collection")
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
    initialize_vector_store(auto_index=AUTO_INDEX_ON_STARTUP)
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

PHRASES_PATH = os.path.join(os.path.dirname(__file__), "settings", "phrases.json")
try:
    with open(PHRASES_PATH, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        if isinstance(loaded, list) and loaded:
            GREETING_PHRASES = loaded
            logger.info("Loaded greeting phrases from phrases.json (list)")
        elif isinstance(loaded, dict) and loaded.get("greeting_phrases"):
            gp = loaded.get("greeting_phrases")
            if isinstance(gp, list) and gp:
                GREETING_PHRASES = gp
                logger.info("Loaded greeting phrases from phrases.json (greeting_phrases)")
            else:
                logger.warning("phrases.json.greeting_phrases is present but not a non-empty list; using defaults")
        else:
            logger.warning("phrases.json loaded but content is not a recognized format; using defaults")
except Exception as e:
    logger.warning(f"Could not load phrases.json: {e}. Using defaults.")

def is_greeting(text: str) -> bool:
    if not text:
        return False
    q = text.lower()
    return any(phrase in q for phrase in GREETING_PHRASES)


def generate_response(question: str, context_docs: list):
    """Generate a response using RAG - returns tuple (answer, used_context)."""
    
    if not context_docs:
        return prompt_config.get("no_context_response", "Nie znalazłem informacji na ten temat w dokumentach."), False
    
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
        return prompt_config.get("no_relevant_docs_response", "Nie znalazłem wystarczająco trafnych informacji w dokumentach."), False
    
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
        
        return answer, True
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return prompt_config.get("error_response", "Przepraszam, wystąpił błąd podczas generowania odpowiedzi."), False


def user_requests_ticket(question: str) -> bool:
    """Detect whether the user explicitly wants to create a BOS ticket."""
    q = question.lower()
    ticket_phrases = [
        "utwórz zgłoszenie", "zrób zgłoszenie", "złóż zgłoszenie", "otwórz zgłoszenie",
        "ticket", "zgłoszenie", "przekaż do bos", "do bos", "biuro obsługi studenta",
        "biuro obsługi", "bos", "bosu", "wyślij zgłoszenie"
    ]
    return any(phrase in q for phrase in ticket_phrases)


def classify_priority(question: str) -> str:
    """Heuristic priority classifier for tickets."""
    q = question.lower()
    high_signals = ["pilne", "asap", "natychmiast", "nie działa", "błąd", "blokada", "deadline", "termin"]
    medium_signals = ["problem", "issue", "opóźnienie", "zwłoka"]
    if any(sig in q for sig in high_signals):
        return "high"
    if any(sig in q for sig in medium_signals):
        return "medium"
    return "low"


def classify_category(question: str) -> str:
    """Lightweight category guess based on keywords."""
    q = question.lower()
    categories = {
        "finance": ["opłata", "płatność", "czesne", "faktura", "rachunek"],
        "it_access": ["logowanie", "hasło", "dostęp", "konto", "loguj"],
        "documents": ["zaświadczenie", "dokument", "podanie", "wniosek"],
        "schedule": ["plan", "zajęć", "terminarz", "harmonogram"],
        "general": []
    }
    for cat, keywords in categories.items():
        if any(word in q for word in keywords):
            return cat
    return "general"


def create_ticket(question: str) -> dict:
    """Create a simple BOS ticket and persist it locally."""
    ticket = {
        "id": str(uuid.uuid4()),
        "description": question,
        "priority": classify_priority(question),
        "category": classify_category(question),
        "status": "new",
        "source": "agent2_ticket",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }
    try:
        if os.path.exists(TICKET_STORE_PATH):
            with open(TICKET_STORE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    data = []
        else:
            data = []
        data.append(ticket)
        with open(TICKET_STORE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Created BOS ticket {ticket['id']} with priority {ticket['priority']} and category {ticket['category']}")
    except Exception as e:
        logger.error(f"Failed to persist ticket: {e}")
    return ticket


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
    # If the user only greets, reply with a short assistant introduction
    if is_greeting(question):
        assistant_name = behavior.get("assistant_name", "Asystent")
        assistant_role = behavior.get("assistant_role", "")
        intro = f"Cześć, jestem {assistant_name} - {assistant_role}. Jak mogę pomóc?"
        return {
            "response": intro,
            "collection": COLLECTION,
            "sources_found": 0
        }
    
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
    
    response, used_context = generate_response(question, context_docs)
    if (not used_context) and user_requests_ticket(question):
        ticket = create_ticket(question)
        ticket_msg = (
            "Nie znalazłem dokładnej odpowiedzi, więc utworzyłem zgłoszenie do BOS. "
            f"ID: {ticket['id']}, kategoria: {ticket['category']}, priorytet: {ticket['priority']}."
        )
        return {
            "response": ticket_msg,
            "collection": COLLECTION,
            "sources_found": len(context_docs),
            "ticket": ticket,
        }
    
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
        # Delete existing collection and re-index documents explicitly
        qdrant_client.delete_collection(COLLECTION)
        logger.info(f"Deleted collection: {COLLECTION}")

        # Recreate collection and force indexing now
        initialize_vector_store(auto_index=True)

        return {
            "status": "success",
            "message": "Documents reloaded and indexed successfully",
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