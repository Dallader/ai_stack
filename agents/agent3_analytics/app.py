import os
from fastapi import FastAPI
from langchain_community.chat_models import ChatOllama

app = FastAPI()
llm = ChatOllama(model="tinyllama", base_url="http://ollama:11434")

COLLECTION = os.getenv("COLLECTION", "agent3_stats")

@app.post("/run")
async def run(payload: dict):
    data = payload.get("input", "")
    response = llm.invoke(f"Analyze this data and provide insights: {data}")
    return {"insights": response.content, "collection": COLLECTION}
