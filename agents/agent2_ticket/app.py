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
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

COLLECTION = os.getenv("COLLECTION", "agent2_tickets")
TICKET_COLLECTION = os.getenv("TICKET_COLLECTION", "tickets")
QDRANT_HOST = os.getenv("QDRANT_HOST", "qdrant")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH", "/app/documents")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:latest")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))
LLM_NUM_PREDICT = int(os.getenv("LLM_NUM_PREDICT", "512"))
FORCE_RECREATE_COLLECTION = (
    os.getenv("FORCE_RECREATE_COLLECTION", "false").lower() == "true"
)
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
ticket_vector_store = None

conversations = {}

prompt_config = {}
PROMPT_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "settings", "prompt_config.json"
)
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
        "error_response": "Przepraszam, wystąpił błąd.",
    }

behavior = {}
BEHAVIOR_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__), "settings", "behavior.json"
)
try:
    with open(BEHAVIOR_CONFIG_PATH, "r", encoding="utf-8") as f:
        behavior = json.load(f)
        logger.info("Loaded behavior configuration from behavior.json")
except Exception as e:
    logger.warning(f"Could not load behavior config: {e}. Using defaults.")
    behavior = {
        "assistant_name": "Asystent WSB Merito",
        "assistant_role": "helper dla studentów WSB Merito",
        "language": prompt_config.get("language", "pl"),
    }


def load_documents_from_folder(folder_path: str):
    """Load all supported documents from a folder and subfolders, adding category metadata."""
    documents = []

    patterns = {
        "**/*.pdf": PyPDFLoader,
        "**/*.txt": TextLoader,
        "**/*.docx": UnstructuredWordDocumentLoader,
        "**/*.doc": UnstructuredWordDocumentLoader,
        "**/*.xlsx": UnstructuredExcelLoader,
        "**/*.xls": UnstructuredExcelLoader,
    }

    for pattern, loader_class in patterns.items():
        file_paths = glob.glob(os.path.join(folder_path, pattern), recursive=True)
        for file_path in file_paths:
            try:
                logger.info(f"Loading document: {file_path}")
                loader = loader_class(file_path)
                docs = loader.load()

                # Determine category from relative path
                rel_path = os.path.relpath(file_path, folder_path)
                category = os.path.dirname(rel_path)
                if not category or category == ".":
                    category = "general"

                for doc in docs:
                    doc.metadata["source_file"] = os.path.basename(file_path)
                    doc.metadata["category"] = category
                documents.extend(docs)
                logger.info(
                    f"Successfully loaded {len(docs)} pages from {file_path} in category {category}"
                )
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

    return documents


def initialize_vector_store(auto_index: bool = False):
    """Initialize Qdrant vector stores for documents and tickets."""
    global vector_store, ticket_vector_store

    try:
        collections = qdrant_client.get_collections().collections
        collection_names = [c.name for c in collections]

        if FORCE_RECREATE_COLLECTION and auto_index:
            try:
                qdrant_client.delete_collection(COLLECTION)
                logger.info(f"Force deleted collection: {COLLECTION}")
            except Exception as e:
                logger.warning(f"Force delete collection failed or not present: {e}")

        if COLLECTION not in collection_names:
            logger.info(f"Creating new collection: {COLLECTION}")
            qdrant_client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM, distance=Distance.COSINE
                ),
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

        if TICKET_COLLECTION not in collection_names:
            logger.info(f"Creating new ticket collection: {TICKET_COLLECTION}")
            qdrant_client.create_collection(
                collection_name=TICKET_COLLECTION,
                vectors_config=VectorParams(
                    size=EMBEDDING_DIM, distance=Distance.COSINE
                ),
            )
            ticket_vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=TICKET_COLLECTION,
                embedding=embeddings,
            )
        else:
            logger.info(
                f"Ticket collection {TICKET_COLLECTION} already exists, connecting..."
            )
            ticket_vector_store = QdrantVectorStore(
                client=qdrant_client,
                collection_name=TICKET_COLLECTION,
                embedding=embeddings,
            )

            collection_info = qdrant_client.get_collection(TICKET_COLLECTION)
            logger.info(f"Ticket collection has {collection_info.points_count} vectors")

    except Exception as e:
        logger.error(f"Error initializing vector stores: {e}")
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
    """Search for relevant documents and tickets in Qdrant."""
    all_results = []

    if vector_store is not None:
        try:
            doc_results = vector_store.similarity_search_with_score(query, k=k)
            for doc, score in doc_results:
                doc.metadata["source"] = "documents"
            all_results.extend(doc_results)
            logger.info(f"Found {len(doc_results)} document results for query")
        except Exception as e:
            logger.error(f"Error searching documents: {e}")

    if ticket_vector_store is not None:
        try:
            ticket_results = ticket_vector_store.similarity_search_with_score(
                query, k=k
            )
            for doc, score in ticket_results:
                doc.metadata["source"] = "tickets"
            all_results.extend(ticket_results)
            logger.info(f"Found {len(ticket_results)} ticket results for query")
        except Exception as e:
            logger.error(f"Error searching tickets: {e}")

    all_results.sort(key=lambda x: x[1])

    return all_results[:k]


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
                logger.info(
                    "Loaded greeting phrases from phrases.json (greeting_phrases)"
                )
            else:
                logger.warning(
                    "phrases.json.greeting_phrases is present but not a non-empty list; using defaults"
                )
        else:
            logger.warning(
                "phrases.json loaded but content is not a recognized format; using defaults"
            )
except Exception as e:
    logger.warning(f"Could not load phrases.json: {e}. Using defaults.")


def is_greeting(text: str) -> bool:
    if not text:
        return False
    q = text.lower()
    return any(phrase in q for phrase in GREETING_PHRASES)


def generate_response(
    question: str, context_docs: list, conversation_history: list = None
):
    """Generate a response using RAG - returns tuple (answer, used_context, needs_clarification, propose_ticket)."""

    conversation_history = conversation_history or []

    if not context_docs:
        return (
            prompt_config.get(
                "no_context_response",
                "Nie znalazłem informacji na ten temat w dokumentach.",
            ),
            False,
            True,
            False,
        )

    clarified_question = question
    student_phrases = prompt_config.get("student_deadline_phrases", ["ile mam"])
    if any(phrase in question.lower() for phrase in student_phrases):
        clarification_prefix = prompt_config.get(
            "clarification_prefix", "PYTANIE STUDENTA:"
        )
        clarified_question = f"{clarification_prefix} {question}"
        logger.info(
            f"Clarified question as student deadline query: {clarified_question}"
        )

    for doc, score in context_docs:
        logger.info(
            f"Document score: {score}, source: {doc.metadata.get('source_file', 'unknown')}"
        )

    relevant_docs = [(doc, score) for doc, score in context_docs if score < 0.9]

    if not relevant_docs:
        logger.info(f"No relevant documents found (best score: {context_docs[0][1]})")
        if len(conversation_history) > 2:
            return (
                prompt_config.get(
                    "propose_ticket",
                    "Nie mogę znaleźć odpowiedzi. Czy chcesz zgłoszenie do BOS?",
                ),
                False,
                False,
                True,
            )
        else:
            return (
                prompt_config.get(
                    "ask_for_details", "Czy możesz podać więcej szczegółów?"
                ),
                False,
                True,
                False,
            )

    context_parts = []

    for doc, score in relevant_docs[:3]:
        logger.info(f"Using document with score: {score}")
        category = doc.metadata.get("category", "unknown")
        context_parts.append(f"Kategoria: {category}\n{doc.page_content}")

    context = "\n\n".join(context_parts)

    history_text = ""
    if conversation_history:
        history_text = (
            "HISTORIA ROZMOWY:\n"
            + "\n".join(
                [
                    f"Student: {msg['question']}\nAsystent: {msg['response']}"
                    for msg in conversation_history[-4:]
                ]
            )
            + "\n\n"
        )

    system_instruction = prompt_config.get("system_instruction", "Jesteś asystentem.")
    critical_rules = prompt_config.get("critical_rules", [])
    response_instruction = prompt_config.get("response_instruction", "ODPOWIEDŹ:")

    rules_text = "\n".join([f"{i+1}. {rule}" for i, rule in enumerate(critical_rules)])

    prompt = f"""{system_instruction}

{history_text}KONTEKST Z DOKUMENTÓW:
{context}

AKTUALNE PYTANIE: {clarified_question}

{response_instruction}"""

    if rules_text:
        prompt += f"\n\nKRYTYCZNE ZASADY - CZYTAJ UWAŻNIE:\n{rules_text}"

    try:
        response = llm.invoke(prompt)
        answer = response.content.strip()

        # Filter out unwanted lines
        lines = answer.split("\n")
        filtered_lines = []
        for line in lines:
            line_lower = line.lower()
            if not any(
                forbidden in line_lower
                for forbidden in [
                    "zasady",
                    "student (nie uczelnia)",
                    "krytyczne zasady",
                    "nie ujawniaj",
                ]
            ):
                filtered_lines.append(line)
        answer = "\n".join(filtered_lines).strip()

        needs_clarification = any(
            phrase in answer.lower()
            for phrase in [
                "więcej szczegółów",
                "więcej informacji",
                "dokładniej",
                "konkretniej",
            ]
        )
        propose_ticket = any(
            phrase in answer.lower()
            for phrase in ["zgłoszenie", "bos", "biuro obsługi", "nie mogę pomóc"]
        )

        return answer, True, needs_clarification, propose_ticket
    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return (
            prompt_config.get(
                "error_response",
                "Przepraszam, wystąpił błąd podczas generowania odpowiedzi.",
            ),
            False,
            False,
            False,
        )


def user_requests_ticket(question: str) -> bool:
    """Detect whether the user explicitly wants to create a BOS ticket."""
    q = question.lower()
    ticket_phrases = [
        "utwórz zgłoszenie",
        "zrób zgłoszenie",
        "złóż zgłoszenie",
        "otwórz zgłoszenie",
        "ticket",
        "zgłoszenie",
        "przekaż do bos",
        "do bos",
        "biuro obsługi studenta",
        "biuro obsługi",
        "bos",
        "bosu",
        "wyślij zgłoszenie",
    ]
    return any(phrase in q for phrase in ticket_phrases)


def classify_priority(question: str) -> str:
    """Heuristic priority classifier for tickets."""
    q = question.lower()
    high_signals = [
        "pilne",
        "asap",
        "natychmiast",
        "nie działa",
        "błąd",
        "blokada",
        "deadline",
        "termin",
    ]
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
        "general": [],
    }
    for cat, keywords in categories.items():
        if any(word in q for word in keywords):
            return cat
    return "general"


def collect_student_data(conversation_history: list) -> dict:
    """Extract student data from conversation history."""
    data = {"name": "", "surname": "", "email": "", "index_number": ""}
    for msg in conversation_history:
        text = msg.get("question", "").lower()
        if "imię" in text or "imie" in text:
            pass
    return data


def create_bos_ticket(
    question: str, conversation_history: list, student_data: dict
) -> dict:
    """Create a BOS ticket with full conversation and student data."""
    # Build structured conversation array
    conversation_struct = []
    for msg in conversation_history:
        if msg.get("question"):
            conversation_struct.append({"student": msg["question"]})
        if msg.get("response"):
            conversation_struct.append({"agent": msg["response"]})

    # Determine ticket category based on the entire conversation context
    full_text = " ".join(
        [
            (msg.get("question") or "") + " " + (msg.get("response") or "")
            for msg in conversation_history
        ]
    )
    ticket_category = classify_category(full_text)

    ticket = {
        "id": str(uuid.uuid4()),
        "subject": "reklamacja oceny",  # For this use case, subject is fixed
        "student_details": {
            "name": student_data.get("name", ""),
            "surname": student_data.get("surname", ""),
            "email": student_data.get("email", ""),
            "index_id": student_data.get("index_number", ""),
        },
        "conversation": conversation_struct,
        "priority": classify_priority(question),
        "category": ticket_category,
        "status": "new",
        "source": "agent2_ticket",
        "created_at": datetime.utcnow().isoformat() + "Z",
    }

    try:
        if ticket_vector_store:
            ticket_text = f"Ticket: {ticket['subject']}"
            ticket_vector_store.add_texts(
                [ticket_text],
                metadatas=[
                    {
                        "type": "ticket",
                        "ticket_id": ticket["id"],
                        "status": ticket["status"],
                        "priority": ticket["priority"],
                        "category": ticket["category"],
                        "subject": ticket["subject"],
                        "student_details": ticket["student_details"],
                        "conversation": ticket["conversation"],
                        "created_at": ticket["created_at"],
                        "student_name": ticket["student_details"].get("name", ""),
                        "student_surname": ticket["student_details"].get("surname", ""),
                        "student_email": ticket["student_details"].get("email", ""),
                        "student_index_id": ticket["student_details"].get(
                            "index_id", ""
                        ),
                    }
                ],
            )
            logger.info(
                f"Saved ticket {ticket['id']} to Qdrant ticket collection (structured, with student details)"
            )
    except Exception as e:
        logger.error(f"Failed to save ticket to Qdrant: {e}")

    logger.info(
        f"Created BOS ticket {ticket['id']} with priority {ticket['priority']} and category {ticket['category']}"
    )
    return ticket


@app.post("/run")
async def run(payload: dict):
    """Main endpoint for processing student questions."""
    question = payload.get("input", "")
    session_id = payload.get("session_id", str(uuid.uuid4()))

    if not question.strip():
        return {
            "response": "Proszę zadaj pytanie.",
            "collection": COLLECTION,
            "sources_found": 0,
            "session_id": session_id,
        }

    logger.info(f"Received question: {question} for session {session_id}")

    if session_id not in conversations:
        conversations[session_id] = {
            "history": [],
            "state": "normal",
            "student_data": {},
        }

    conv = conversations[session_id]
    conversation_history = conv["history"]
    state = conv["state"]
    student_data = conv["student_data"]

    if is_greeting(question):
        assistant_name = behavior.get("assistant_name", "Asystent")
        assistant_role = behavior.get("assistant_role", "")
        intro = f"Cześć, jestem {assistant_name} - {assistant_role}. Jak mogę pomóc?"
        conversation_history.append({"question": question, "response": intro})
        return {
            "response": intro,
            "collection": COLLECTION,
            "sources_found": 0,
            "session_id": session_id,
        }

    doc_list_phrases = [
        "jakie masz dokumenty",
        "jakie dokumenty",
        "lista dokumentów",
        "pokaż dokumenty",
        "what documents",
    ]
    if any(phrase in question.lower() for phrase in doc_list_phrases):
        documents = []
        for pattern in ["*.pdf", "*.txt"]:
            files = glob.glob(os.path.join(DOCUMENTS_PATH, pattern))
            documents.extend([os.path.basename(f) for f in files])

        doc_list = "\n".join([f"- {doc}" for doc in documents])
        response = f"Mam dostęp do następujących dokumentów:\n{doc_list}"
        conversation_history.append({"question": question, "response": response})
        return {
            "response": response,
            "collection": COLLECTION,
            "sources_found": len(documents),
            "session_id": session_id,
        }

    # Only show previous tickets if explicitly asked
    show_ticket_phrases = [
        "pokaż zgłoszenie",
        "pokaz zgłoszenie",
        "ostatnie zgłoszenie",
        "pokaż zgłoszenia",
        "lista zgłoszeń",
        "historia zgłoszeń",
    ]
    if any(phrase in question.lower() for phrase in show_ticket_phrases):
        ticket_results = []
        if ticket_vector_store is not None:
            try:
                ticket_results = ticket_vector_store.similarity_search_with_score(
                    question, k=3
                )
            except Exception as e:
                logger.error(f"Error searching tickets: {e}")
        if ticket_results and ticket_results[0][1] < 0.8:
            ticket_doc = ticket_results[0][0]
            meta = ticket_doc.metadata
            # Pretty print ticket details
            details = [
                f"Zgłoszenie zostało utworzone!",
                f"ID: {meta.get('ticket_id', '-')}",
                f"Temat: {meta.get('subject', '-')}",
                f"Kategoria: {meta.get('category', '-')}",
                f"Priorytet: {meta.get('priority', '-')}",
                f"Status: {meta.get('status', '-')}",
                f"Data utworzenia: {meta.get('created_at', '-')}",
                "Dane studenta:",
                f"  Imię: {meta.get('student_name', '-')}",
                f"  Nazwisko: {meta.get('student_surname', '-')}",
                f"  Email: {meta.get('student_email', '-')}",
                f"  Nr indeksu: {meta.get('student_index_id', '-')}",
                "Rozmowa:",
            ]
            # Show conversation nicely
            conversation = meta.get("conversation", [])
            for turn in conversation:
                if "student" in turn:
                    details.append(f"Student: {turn['student']}")
                if "agent" in turn:
                    details.append(f"Asystent: {turn['agent']}")
            response = "\n".join(details)
        else:
            response = "Nie znalazłem żadnego zgłoszenia."
        conversation_history.append({"question": question, "response": response})
        return {
            "response": response,
            "collection": COLLECTION,
            "sources_found": len(ticket_results) if ticket_results else 0,
            "session_id": session_id,
        }

    # If user requests a ticket, always start new ticket flow
    if user_requests_ticket(question):
        conv["state"] = "collecting_name"
        response = "Świetnie, utworzę zgłoszenie. Proszę podaj swoje imię."
        conversation_history.append({"question": question, "response": response})
        return {
            "response": response,
            "collection": COLLECTION,
            "sources_found": 0,
            "session_id": session_id,
        }

    context_docs = search_documents(question, k=5)

    if state == "awaiting_ticket_confirmation":
        if "tak" in question.lower() or "zgoda" in question.lower():
            conv["state"] = "collecting_data"
            response = "Świetnie, utworzę zgłoszenie. Proszę podaj swoje dane: imię, nazwisko, email i numer indeksu."
            conversation_history.append({"question": question, "response": response})
            return {
                "response": response,
                "collection": COLLECTION,
                "sources_found": len(context_docs),
                "session_id": session_id,
            }
        else:
            conv["state"] = "normal"
            response = "Rozumiem. Czy mogę pomóc w czymś innym?"
            conversation_history.append({"question": question, "response": response})
            return {
                "response": response,
                "collection": COLLECTION,
                "sources_found": len(context_docs),
                "session_id": session_id,
            }

    if state == "awaiting_ticket_confirmation":
        if (
            "tak" in question.lower()
            or "zgoda" in question.lower()
            or "utwórz" in question.lower()
            or "nowe" in question.lower()
        ):
            conv["state"] = "collecting_name"
            response = "Świetnie, utworzę zgłoszenie. Proszę podaj swoje imię."
            conversation_history.append({"question": question, "response": response})
            return {
                "response": response,
                "collection": COLLECTION,
                "sources_found": len(context_docs),
                "session_id": session_id,
            }
        else:
            conv["state"] = "normal"
            response = "Rozumiem. Czy mogę pomóc w czymś innym?"
            conversation_history.append({"question": question, "response": response})
            return {
                "response": response,
                "collection": COLLECTION,
                "sources_found": len(context_docs),
                "session_id": session_id,
            }

    elif state in [
        "collecting_name",
        "collecting_surname",
        "collecting_email",
        "collecting_index",
        "collecting_new_address",
        "collecting_new_name",
        "collecting_new_surname",
        "collecting_new_bank_account",
    ]:
        required_fields = get_required_fields_for_ticket(question, conversation_history)
        # Determine which field to collect next
        for field in required_fields:
            field_state = f"collecting_{field}"
            if state == field_state:
                student_data[field] = question.strip()
                # Move to next field or finish
                next_idx = required_fields.index(field) + 1
                if next_idx < len(required_fields):
                    next_field = required_fields[next_idx]
                    conv["state"] = f"collecting_{next_field}"
                    prompts = {
                        "new_address": "Proszę podaj nowy adres.",
                        "new_name": "Proszę podaj nowe imię.",
                        "new_surname": "Proszę podaj nowe nazwisko.",
                        "new_bank_account": "Proszę podaj nowy numer konta bankowego.",
                        "name": "Proszę podaj swoje imię.",
                        "surname": "Proszę podaj swoje nazwisko.",
                        "email": "Proszę podaj swój email.",
                        "index_number": "Proszę podaj swój numer indeksu.",
                    }
                    response = prompts.get(next_field, f"Proszę podaj {next_field}.")
                else:
                    # All required fields collected
                    ticket = create_bos_ticket(
                        question, conversation_history, student_data
                    )
                    ticket_msg = "Twoje zgłoszenie zostało utworzone i przekazane do Biura Obsługi Studenta. Dziękujemy! Skontaktujemy się z Tobą wkrótce."
                    conv["state"] = "normal"
                    conversation_history.append(
                        {"question": question, "response": ticket_msg}
                    )
                    return {
                        "response": ticket_msg,
                        "collection": COLLECTION,
                        "sources_found": len(context_docs),
                        "session_id": session_id,
                        "ticket": ticket,
                    }
                conversation_history.append(
                    {"question": question, "response": response}
                )
                return {
                    "response": response,
                    "collection": COLLECTION,
                    "sources_found": len(context_docs),
                    "session_id": session_id,
                }
        # fallback for unknown state
        response = "Brakuje danych. Spróbuj ponownie od początku."
        conv["state"] = "normal"
        conversation_history.append({"question": question, "response": response})
        return {
            "response": response,
            "collection": COLLECTION,
            "sources_found": len(context_docs),
            "session_id": session_id,
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
        "document_collection": COLLECTION,
        "ticket_collection": TICKET_COLLECTION,
        "vector_store_ready": vector_store is not None,
        "ticket_vector_store_ready": ticket_vector_store is not None,
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

        initialize_vector_store(auto_index=True)

        return {
            "status": "success",
            "message": "Documents reloaded and indexed successfully",
            "collection": COLLECTION,
        }
    except Exception as e:
        logger.error(f"Error reloading documents: {e}")
        return {"status": "error", "message": str(e)}


@app.get("/documents")
async def list_documents():
    """List all documents in the documents folder."""
    documents = []
    for pattern in ["*.pdf", "*.txt"]:
        files = glob.glob(os.path.join(DOCUMENTS_PATH, pattern))
        documents.extend([os.path.basename(f) for f in files])

    return {"documents": documents, "count": len(documents), "path": DOCUMENTS_PATH}
