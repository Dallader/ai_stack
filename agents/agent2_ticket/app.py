# Imports
import os, base64
import streamlit as st
import json
# OpenaAI API
from openai import OpenAI
# Load env's
from dotenv import load_dotenv
from typing import List, Dict, Any
from pathlib import Path
from dataclasses import dataclass
# Qdrant
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct
# Tooling
from tools.tools import *

# Debug check
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "yes")

if DEBUG:
    # Directories
    BASE_DIR = Path(__file__).parent
    ## CSS file directory
    CSS_FILE = BASE_DIR / "static" / "css" / "main.css"
    ## Documents directories
    DOCS_DIR = BASE_DIR / "documents"
    KNOWLEDGE_DIR = DOCS_DIR / "knowledge"
    LOGS_DIR = DOCS_DIR / "logs"
    UPLOADS_DIR = DOCS_DIR / "uploads"
    PROCESSED_DIR = DOCS_DIR / "processed"
    ## Settings directory (json files)
    SETTINGS_DIR = BASE_DIR / "settings"
else:
    # Use container-mounted paths or environment variables
    SETTINGS_DIR = "/app/settings"
    KNOWLEDGE_DIR = "/app/documents/knowledge"
    LOGS_DIR = "/app/documents/logs"
    UPLOADS_DIR = "/app/documents/uploads"
    PROCESSED_DIR = "/app/documents/processed"
    CSS_FILE =  "/app/static/css/main.css"

# Read .env file and set the variables
load_dotenv()

# Retrive the credentials
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# VECTOR_STORE_ID = os.getenv("VECTOR_STORE_ID") or st.secrets["VECTOR_STORE_ID"]
MODEL_NAME = os.getenv("MODEL_NAME")
CLASSIFYING_MODEL_NAME = os.getenv("CLASSIFYING_MODEL_NAME")
EMBEDING_MODEL_NAME = os.getenv("EMBEDING_MODEL_NAME")
#QDRANT_URL = os.getenv("QDRANT_URL") or st.secrets["QDRANT_URL"]
#QDRANT_API_KEY = os.getenv("QDRANT_API_KEY") or st.secrets["QDRANT_API_KEY"]
# Sidebar check
SIDEBAR = os.getenv("SIDEBAR", "False").lower() in ("true", "1", "yes")

# Set the OpenAI API key in the os
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# URLs
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")

# Initialize Qdrant
#qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL)
# Ensure required collections exist
#collections_to_check = ["Documents", "Knowledge", "Processed", "Tickets"]
collections_to_check = ["Documents", "Tickets"]
result = ensure_collections_exist(qdrant_client, collections_to_check)

# Load css file to override streamlit styles
@st.cache_data
def get_css(css_path: Path):
    with open(css_path) as f:
        return f.read()

st.markdown(f"<style>{get_css(CSS_FILE)}</style>", unsafe_allow_html=True)

# Load JSON files to objects
## -> LLM
with open(f"{SETTINGS_DIR}/llm.json", "r", encoding="utf-8") as file:
    llm_settings = json.load(file)
## System Prompt
system_prompt = build_system_prompt(llm_settings["system_prompt"])

## -> Qdrant
QDRANT_COLLECTION_DOCS = "Documents"
QDRANT_COLLECTION_TICKETS = "Tickets"

## -> Categories
with open(f"{SETTINGS_DIR}/categories.json", "r", encoding="utf-8") as file:
    categories_list = json.load(file)

@dataclass
class Category:
    name: str
category_objects = [Category(name) for name in categories_list["categories"]]

# Get knowledge 
knowledge_not_imported = not_imported_files(qdrant_client, KNOWLEDGE_DIR) 
# Get Qdrant collections information
#qdrant_collections_info = get_qdrant_collection_summary(QDRANT_URL, QDRANT_API_KEY)
qdrant_collections_info = get_qdrant_collection_summary(QDRANT_URL)

# App configurations
st.set_page_config(
    page_title="AI RAG Agent",
    page_icon=":material/chat_bubble:", # speech bubble icon
    layout="centered"
)

# Add title to the app
st.title("WSB RAG Agent")

# Add a description to the ap
st.markdown("**Your inteligent WSB Assistant powered by GPT-5 and RAG technology**")
st.divider()

# Add a collapsible section
with st.expander("About this chat", expanded=False):
    st.markdown(
        """
        ### Intelligent Assistant (Chat & Ticketing)
        - Model
            - **GPT-5 class** (configurable via `MODEL_NAME`) using **OpenAI Responses API**
        - Retrieval (RAG)
            - **Qdrant vector search** (top-k context from `Documents` collection) to ground answers in your docs.
        - Features
            - Multi-turn conversational chat
            - Document & image input
            - Clear / reset conversation
        - Secrets & Configuration (env or Streamlit secrets):
            - `OPENAI_API_KEY`
            - `MODEL_NAME`
            - `EMBEDING_MODEL_NAME`
            - `QDRANT_URL`
            - `QDRANT_API_KEY`
        ---
        - How it works
            1. Your message (plus optional uploads) is sent to the **Responses API**
            2. Relevant passages are fetched from **Qdrant** using embeddings from `EMBEDING_MODEL_NAME`
            3. The model blends the retrieved context to return grounded, concise answers
        """
    )

# Initialize the OpenAi Client
client = OpenAI()

# Warn if OpenAI API Key or the VectorStoreId are not set
if not OPENAI_API_KEY:
    st.warning("OpenAI API Key is not set. Please set OPENAI_API_KEY in the environment")
if not MODEL_NAME:
    st.warning("Model name is not set. Please set MODEL_NAME in the environment")
if not EMBEDING_MODEL_NAME:
    st.warning("Embedding model name is not set. Please set EMBEDING_MODEL_NAME in the environment")
#if not QDRANT_URL or not QDRANT_API_KEY:
if not QDRANT_URL:
    st.warning("Qdrant connection is not set. Please set QDRANT_URL and QDRANT_API_KEY in the environment")

# Store the previous response id
if "previous_response_id" not in st.session_state:
    st.session_state.previous_response_id = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# User sidebar
with st.sidebar:
    st.header("User Controls")

    # Clear the converstation history - reset chat history and context
    if st.button("Clear Conversation History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.previous_response_id = None
        # Reest the page
        st.rerun()

    if SIDEBAR:
        st.divider() 

        # Qdrant Management
        st.subheader("Qdrant Management")
        # Compute not imported files at button click
        knowledge_not_imported = not_imported_files(qdrant_client, KNOWLEDGE_DIR)
        st.write(f"Files to import: {knowledge_not_imported}")
        
        if st.button(f"Import Documents"):
            import_and_index_documents_qdrant(qdrant_client, client, EMBEDING_MODEL_NAME, KNOWLEDGE_DIR, CLASSIFYING_MODEL_NAME, SETTINGS_DIR)

        # Display each collection in Streamlit
        for coll in qdrant_collections_info:
            st.markdown(f"**Collection:** {coll['name']}")
            st.markdown(f"- Status: {coll['status']}")
            st.markdown(f"- Points: {coll['points']}")
            #t.markdown(f"- Segments: {coll['shards']}")
            #st.markdown(f"- Replicas: {coll['replicas']}")
            #st.markdown(f"- Vector Field: {coll['vector_field']}")
            #st.markdown(f"- Vector Size: {coll['vector_size']}")
            #st.markdown(f"- Distance: {coll['distance']}")
            st.divider()
            
        st.title("Qdrant Category Viewer")
        if st.button("Get Category Counts"):
            st.info("Fetching categories from Qdrant...")
            category_counts = get_category_counts(qdrant_client, "Documents")

            if category_counts:
                st.success(f"Found {sum(category_counts.values())} points in {len(category_counts)} categories.")
                for category, count in category_counts.items():
                    st.write(f"**{category}:** {count} points")
            else:
                st.warning("No categories found in the collection.")

# Render all previous messages
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        # extract content from the message structure
        if isinstance(m["content"], list):
            for part in m["content"]:
                for content_item in part.get("content", []):
                    if content_item.get("type") == "input_text":
                        st.markdown(content_item["text"])
                    elif content_item.get("type") == "input_image":
                        st.image(content_item["image_url"], width=100)
        elif isinstance(m["content"], str):
            st.markdown(m["content"])

# User Interface
## Upload files
uploaded = st.file_uploader(
    "Upload file(s)",
    type=["jpg","jpeg","png","webp","doc","docx","xls","xlsx","txt","pdf"],
    accept_multiple_files=True,
    key=f"file_uploader_{len(st.session_state.messages)}"
)

# Zapis wszystkich uploadów w sesji
if uploaded:
    if "uploaded_files" not in st.session_state:
        st.session_state["uploaded_files"] = []
    st.session_state["uploaded_files"].extend(uploaded)

## Chat input
prompt = st.chat_input("Type your message here..")

if "collecting_student_data" not in st.session_state:
    st.session_state.collecting_student_data = False

if "missing_fields" not in st.session_state:
    st.session_state.missing_fields = []

if prompt is not None:
    
    if st.session_state.get("collecting_student_data"):
        field = st.session_state.get("current_field")

        if field:
            st.session_state[f"user_{field}"] = prompt

            # usuń uzupełnione pole z listy brakujących
            st.session_state.missing_fields.remove(field)

            if st.session_state.missing_fields:
                # pytamy o kolejne pole
                next_field = st.session_state.missing_fields[0]
                st.session_state.current_field = next_field

                question_map = {
                    "first_name": "Podaj swoje imię:",
                    "last_name": "Podaj swoje nazwisko:",
                    "email": "Podaj swój adres email:",
                    "index_number": "Podaj swój numer indeksu:"
                }

                st.chat_message("assistant").markdown(question_map[next_field])
                st.stop()
            else:
                # mamy komplet danych
                st.session_state.collecting_student_data = False
                st.success("Dziękuję. Mam już wszystkie dane. Tworzę zgłoszenie...")
                st.rerun()
                
    # Handle uploaded files only for this user message
    images = []
    documents = []

    for f in uploaded or []:
        # Save file to uploads folder 
        saved_path = save_uploaded_file(f, UPLOADS_DIR)
        suffix = saved_path.suffix.lower()

        if suffix in [".jpg", ".jpeg", ".png", ".webp"]:
            # Encode images for chat display
            with open(saved_path, "rb") as img_f:
                data_url = f"data:image/{suffix[1:]};base64,{base64.b64encode(img_f.read()).decode('utf-8')}"
            images.append({"mime_type": f"image/{suffix[1:]}", "data_url": data_url})
        else:
            documents.append(saved_path)

    # Build the input parts for the responses API (text + images)
    parts = build_input_parts(prompt, images)

    # Store the user's message in chat session
    st.session_state.messages.append({"role": "user", "content": parts})

    # Display user's message
    with st.chat_message("user"):
        for p in parts:
            if p["type"] == "message":
                for content_item in p.get("content", []):
                    if content_item["type"] == "input_text":
                        st.markdown(content_item["text"])
                    elif content_item["type"] == "input_image":
                        st.image(content_item["image_url"], width=100)

    is_ticket_intent = should_create_ticket(prompt, "")

    if is_ticket_intent:
        change = extract_change_request(client, MODEL_NAME, prompt)

        if not change.get("field") or not change.get("new_value"):
            st.warning(
                "Aby utworzyć zgłoszenie o zmianę danych, podaj:\n"
                "- co chcesz zmienić (np. email)\n"
                "- nową wartość\n\n"
                "Przykład: *Chcę zmienić email na nowy@email.pl*"
            )
            st.stop()

        ticket_data = {
            "first_name": st.session_state.get("user_first_name"),
            "last_name": st.session_state.get("user_last_name"),
            "email": st.session_state.get("user_email"),
            "index_number": st.session_state.get("user_index"),
            "description": (
                f"Prośba o zmianę danych.\n"
                f"Pole: {change['field']}\n"
                f"Stara wartość: {change.get('old_value') or 'nie podano'}\n"
                f"Nowa wartość: {change['new_value']}"
            )
        }

        missing = [k for k, v in ticket_data.items() if not v]

        if missing:
            st.session_state.collecting_student_data = True
            st.session_state.missing_fields = missing

            question_map = {
                "first_name": "Podaj swoje imię:",
                "last_name": "Podaj swoje nazwisko:",
                "email": "Podaj swój adres email:",
                "index_number": "Podaj swój numer indeksu:"
            }

            first_missing = missing[0]
            st.session_state.current_field = first_missing

            st.chat_message("assistant").markdown(question_map[first_missing])
            st.stop()
            
        
        ticket_info = interactive_ticket_creation(
            qdrant_client=qdrant_client,
            openai_client=client,
            model_name=MODEL_NAME,
            ticket_data=ticket_data,
            interactive=False
        )

        st.success(f"Ticket utworzony! ID: {ticket_info['ticket_id']}")
        st.info(f"Kategoria: {ticket_info['category']}")
        st.stop()
        
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = call_responses_api(
                    system_prompt=system_prompt,
                    client=client,
                    model_name=MODEL_NAME,
                    embedding_model_name=EMBEDING_MODEL_NAME,
                    parts=parts,
                    qdrant_client=qdrant_client,
                    qdrant_collection=QDRANT_COLLECTION_DOCS,
                    top_k=5,
                    previous_response_id=st.session_state.previous_response_id
                )

                output_text = get_text_output(response)
                st.markdown(output_text)
                st.session_state.messages.append({"role": "assistant", "content": output_text})

                if hasattr(response, "id"):
                    st.session_state.previous_response_id = response.id

            except Exception as e:
                st.error(f"Error generating response: {e}")

    for doc_path in documents:
        text = doc_path.read_text(encoding="utf-8").strip()
        
        if len(text) < 1000:
            st.markdown(f"**Treść dokumentu {doc_path.name}:** {text}")
        else:
            # Qdrant processing & embedding
            import_and_index_documents_qdrant(
                qdrant_client,
                client,
                EMBEDING_MODEL_NAME,
                doc_path.parent,
                MODEL_NAME,
                SETTINGS_DIR
            )
        
        # Move file to processed and uploaded folder
        move_to_processed(PROCESSED_DIR, doc_path)