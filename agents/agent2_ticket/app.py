import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.chat_models import ChatOllama

app = FastAPI()

# Enable CORS for the chat interface
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOllama(model="llama3", base_url="http://ollama:11434")

COLLECTION = os.getenv("COLLECTION", "agent2_tickets")

@app.post("/run")
async def run(payload: dict):
    ticket_text = payload.get("input", "")
    response = llm.invoke(f"Classify this ticket: {ticket_text}")
    # tu zakładamy JSON output
    return {"ticket_classification": response.content, "collection": COLLECTION}