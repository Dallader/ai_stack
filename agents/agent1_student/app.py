import os
from fastapi import FastAPI
from langchain_community.chat_models import ChatOllama

app = FastAPI()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "tinyllama")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "2048"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "256"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.2"))

llm = ChatOllama(
    model=OLLAMA_MODEL,
    base_url=OLLAMA_BASE_URL,
    num_ctx=OLLAMA_NUM_CTX,
    num_predict=OLLAMA_NUM_PREDICT,
    temperature=OLLAMA_TEMPERATURE
)

COLLECTION = os.getenv("COLLECTION", "agent1_student")

@app.post("/run")
async def run(payload: dict):
    question = payload.get("input", "")
    response = llm.invoke(question)
    return {"result": response.content, "collection": COLLECTION}
