# agent2_ticket – Test Checklist

## Setup
- Rebuild and start: `docker compose build agent2_ticket && docker compose up -d agent2_ticket`
- Ensure Ollama has the model: `curl http://localhost:11434/api/tags | jq` (expect `llama3.1:latest`).

## Health and Info
- Health: `curl http://localhost:8002/health` (status healthy, qdrant/ollama connected).
- Info: `curl http://localhost:8002/api` (shows collection and URLs).

## Indexing
- CLI ingest: `docker compose run --rm agent2_ticket python ingest.py` (prints chunk count).
- API ingest: `curl -X POST http://localhost:8002/index` (returns chunk count).

## Categories
- BOS list: `curl http://localhost:8002/documents/categories` (10 categories expected).

## Document Upload (similarity + categorization)
- Upload: `curl -F "file=@sample.pdf" http://localhost:8002/documents/upload`
  - Expect `status` `existing` (with match score) or `indexed` (with category, doc_id, chunks_indexed).
- Verify files: incoming copy in `agents/agent2_ticket/incoming/`, processed text copy in `processed_documents/`.

## Chat Data Collection & Session
- Open UI: http://localhost:8002
- Provide name → surname → email → index; prompts should advance and persist (localStorage).
- Session check: `curl http://localhost:8002/session/<session_id>` (data_collection_complete true after flow).

## Student Data Retrieval
- Known sample index (e.g., 12345): `curl http://localhost:8002/student/12345`
- In chat, ask: "Jaka jest moja średnia ogólna?" → direct numeric response (no LLM).

## RAG Q&A
- Ask about content present in indexed docs; expect contextual answer.
- If no coverage, expect fallback: "Nie mam wystarczających informacji...".

## Ticket Creation
- In chat: include phrase "utwórz zgłoszenie" → expect ticket ID response.
- List tickets: `curl http://localhost:8002/tickets?limit=5` (shows created ticket).

## Upload-and-Index (compat)
- `curl -F "file=@sample.pdf" http://localhost:8002/upload-and-index` (should mirror /documents/upload behavior).

## Collections Overview
- `curl http://localhost:8002/collections` (documents/tickets/students with counts).

## Troubleshooting
- Logs: `docker compose logs -f agent2_ticket`
- Qdrant reachable: `curl http://localhost:6333/collections`
- Reindex after model changes: rerun `python ingest.py` or POST `/index`.
