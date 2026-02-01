## Agent 2 – Ticket & RAG Assistant 💗

> Pink, fluffy, and ready to help students. A FastAPI + Qdrant + Ollama chat that ingests documents, answers questions with RAG, and can raise BOS-style tickets — all with a cozy frontend.

### 🌸 At a Glance
- Friendly chat UI with silent uploads, session restore, and a clear-file button.
- Auto-ingest of PDF/DOC/DOCX/TXT/MD/images (OCR) and XLS/XLSX; dedupe vs Qdrant before indexing.
- RAG flow: embed → search → ground → generate with Ollama.
- Ticket mode: LLM-based category/priority, stored in Qdrant `tickets`.
- Collections auto-provisioned with vector-size checks.

### 🎀 Folder Map
- [agents/agent2_ticket/app.py](agents/agent2_ticket/app.py) — FastAPI app, endpoints, RAG, ingest, tickets.
- [agents/agent2_ticket/ingest.py](agents/agent2_ticket/ingest.py) — batch ingest helper.
- [agents/agent2_ticket/tools/helpers.py](agents/agent2_ticket/tools/helpers.py) — extraction, chunking, embeddings, Qdrant I/O, ticket writer.
- [agents/agent2_ticket/templates/chat.html](agents/agent2_ticket/templates/chat.html) — chat shell.
- [agents/agent2_ticket/static/js/chat.js](agents/agent2_ticket/static/js/chat.js) — frontend logic (sessions, silent upload, status, sequencing).
- [agents/agent2_ticket/static/css/style.css](agents/agent2_ticket/static/css/style.css) — pastel styling.
- settings/behavior.json, phrases.json, prompt_config.json, rules.json — prompts & tone.
- documents/, knowledge/, incoming/, processed_documents/, logs/ — content, uploads, processed text, and logs.

### ✨ Runtime Story
1) Startup: ensure folders exist; auto-create `documents` collection with proper dim; log URLs.
2) First query: if Qdrant empty, index knowledge/incoming/documents/processed once.
3) Frontend: user picks a file (kept pending); on “Wyślij” the file is silently posted to `/documents/upload`, then the chat request goes to `/run`.
4) Upload: extract text → embed query → search Qdrant (threshold 0.82). If similar, skip index; else categorize via LLM, chunk (800/200), embed, upsert with payload (doc_id, source, path, category, origin, uploaded_by) and save processed copy.
5) Chat: embed question → Qdrant top 5 (threshold 0.2) → craft context → Ollama generate/chat → reply.
6) Ticket intent: capture user info, LLM assigns category/priority, upsert into `tickets` with vector built from question + recent chat.

### 🎯 Endpoints Cheat-Sheet
| Method | Path | Purpose |
| --- | --- | --- |
| GET | / | Chat UI |
| GET | /api | Service info |
| GET | /health | Liveness for UI |
| POST | /run | Main chat (RAG + optional pending upload) |
| POST | /documents/upload | Silent ingest + dedupe + index if new |
| GET | /collections | Qdrant collections with counts |
| POST | /clear_history | Clear chat history (keep user data flags) |
| GET | /session/{session_id} | Restore session |

### 🌷 Models & Embeddings
- Generation: Ollama `OLLAMA_MODEL` (default `mistral:7b`), tries `/api/generate` then `/api/chat`.
- Embeddings: `OLLAMA_EMBED_MODEL` (default `nomic-embed-text`), dim `OLLAMA_EMBED_DIM` (default 768), endpoint `OLLAMA_EMBED_URL`.
- Timeouts: `OLLAMA_REQUEST_TIMEOUT` (default 600s), `LLM_TIMEOUT_SECONDS` (default +60 buffer).

### 🧁 Frontend UX
- Silent uploads (no chat noise); hint line lists allowed types.
- Pending-file badge + “Usuń plik” to clear wrong selection.
- Status sequence: “Szukam…”, “Analizuję…”, “Generuję…”.
- Session restore and history clear without losing user data when already collected.

### 🍬 File Handling
- Extensions: pdf, txt, md, doc, docx, png, jpg, jpeg, webp, gif, bmp, xls, xlsx.
- PDF via pypdf; DOC/DOCX via docx2txt; images via Pillow + optional pytesseract; Excel via pandas; text utf-8/latin-1 fallback.
- Chunking: size 800, overlap 200.

### 🎟️ Ticketing
- Categories: Technical Issue, Academic Question, Administrative, Financial, General Inquiry.
- Priority: Low, Medium, High, Critical.
- Stored in Qdrant `tickets` with vector of question + recent chat; payload includes user data and summary.

### 🌺 Environment
- Qdrant: `QDRANT_URL`, `QDRANT_COLLECTION`, `TICKETS_COLLECTION` (defaults documents/tickets).
- Ollama: `OLLAMA_URL`, `OLLAMA_EMBED_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_EMBED_DIM`.
- Time/limits: `OLLAMA_REQUEST_TIMEOUT`, `LLM_TIMEOUT_SECONDS`, `OLLAMA_NUM_PREDICT`, `OLLAMA_NUM_CTX`.

### 🧪 Quickstart (pink edition)
```bash
# 1) Run the stack (from repo root or this folder)


# 2) Open the chat
open http://localhost:8002/

# 3) Optional: batch ingest (example)
python agents/agent2_ticket/ingest.py --dir agents/agent2_ticket/knowledge
```

```bash
# Upload a doc (silent ingest) via curl
curl -F "file=@sample.docx" http://localhost:8002/documents/upload

# Ask a question (after upload)
curl -X POST http://localhost:8002/run \
	-H "Content-Type: application/json" \
	-d '{"input":"Gdzie wyslac prace dyplomowa?"}'
```

### 🌸 Data & Payloads
- Doc payload: doc_id, source, path, category, origin (knowledge/incoming/documents/processed/student_upload), chunk_id, text, uploaded_by.
- Ticket payload: ticket_id, category, priority, user info, question, chat_history, ticket_text, status, timestamp.

### 🛡️ Notes
- OCR needs pytesseract in the image/host.
- Upload size limits depend on infra config.
- Vector-size drift recreates collections to match embedding dim.

### 💖 FAQ
- Duplicate uploads? Detected by similarity ≥ 0.82, skipped from indexing.
- LLM/embeddings down? HTTP 5xx with friendly text; UI shows connection error.
- Want to reset chat only? Use “Wyczyść” — user data is preserved if already captured.
