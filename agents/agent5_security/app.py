import os
from fastapi import FastAPI
from langchain_community.chat_models import ChatOllama

app = FastAPI()
llm = ChatOllama(model="tinyllama", base_url="http://ollama:11434")

COLLECTION = os.getenv("COLLECTION", "agent5_audit")

@app.post("/run")
async def run(payload: dict):
    text = payload.get("input", "")
    # Anonimizacja / walidacja
    response = llm.invoke(f"Check data compliance and anonymize: {text}")
    return {"audit": response.content, "collection": COLLECTION}
