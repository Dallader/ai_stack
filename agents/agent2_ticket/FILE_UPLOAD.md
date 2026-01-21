# File Upload Guide for Agent 2 Ticket System

This guide explains how to upload PDF, Word, Excel, and other documents to Qdrant for use in the Agent 2 ticket system's RAG (Retrieval-Augmented Generation).

## 📁 Supported File Formats

- **PDF**: `.pdf`
- **Word**: `.docx`, `.doc`
- **Excel**: `.xlsx`, `.xls`
- **Text**: `.txt`, `.md`
- **JSON**: `.json`

## 🚀 Method 1: Using the API (Recommended)

### Upload Single File

```bash
# Upload a PDF
curl -X POST http://localhost:8002/upload \
  -F "file=@/path/to/document.pdf" \
  -F "category=FAQ"

# Upload a Word document
curl -X POST http://localhost:8002/upload \
  -F "file=@/path/to/manual.docx" \
  -F "category=Manual"

# Upload with custom category
curl -X POST http://localhost:8002/upload \
  -F "file=@/path/to/policy.pdf" \
  -F "category=Policy"
```

### Upload Multiple Files

```bash
curl -X POST http://localhost:8002/upload/batch \
  -F "files=@document1.pdf" \
  -F "files=@document2.docx" \
  -F "files=@document3.txt" \
  -F "category=Documentation"
```

### List Uploaded Documents

```bash
curl http://localhost:8002/documents
```

**Response:**
```json
{
  "documents": [
    {
      "filename": "user_manual.pdf",
      "category": "Manual",
      "type": "document",
      "chunks": 15
    },
    {
      "filename": "faq.docx",
      "category": "FAQ",
      "type": "document",
      "chunks": 8
    }
  ],
  "total": 2
}
```

### Delete a Document

```bash
curl -X DELETE http://localhost:8002/documents/user_manual.pdf
```

### Get Collection Statistics

```bash
curl http://localhost:8002/stats
```

**Response:**
```json
{
  "collection_name": "agent2_tickets",
  "total_points": 156,
  "vector_size": 4096,
  "document_types": {
    "document": 145,
    "rag_document": 11
  },
  "categories": {
    "FAQ": 45,
    "Manual": 78,
    "Policy": 22,
    "procedury": 11
  }
}
```

## 🐍 Method 2: Using Python Script

### Upload Single File

```bash
# From container
docker exec -it agent2_ticket python upload_documents.py /path/to/document.pdf -c "FAQ"

# With custom settings
docker exec -it agent2_ticket python upload_documents.py \
  /path/to/document.pdf \
  --category "Manual" \
  --collection agent2_tickets \
  --stats
```

### Upload Entire Directory

```bash
# Upload all files from a directory
docker exec -it agent2_ticket python upload_documents.py \
  /path/to/documents/ \
  --category "Documentation" \
  --recursive

# Non-recursive (only direct files)
docker exec -it agent2_ticket python upload_documents.py \
  /path/to/documents/ \
  --category "FAQ"
```

### From Host Machine

```bash
cd /Users/karolsliwka/Desktop/ai_stack/agents/agent2_ticket

# Upload single file
python3 upload_documents.py \
  /path/to/document.pdf \
  --category "FAQ" \
  --qdrant-url http://localhost:6333 \
  --ollama-url http://localhost:11434

# Upload directory
python3 upload_documents.py \
  /path/to/documents/ \
  --category "Documentation" \
  --recursive \
  --stats
```

## 🐍 Method 3: Using Python Code

```python
from file_uploader import DocumentUploader, upload_file_to_qdrant

# Quick upload
result = upload_file_to_qdrant(
    file_path="/path/to/document.pdf",
    category="FAQ"
)
print(result)

# Advanced usage
uploader = DocumentUploader(
    qdrant_url="http://localhost:6333",
    collection_name="agent2_tickets",
    ollama_url="http://localhost:11434",
    chunk_size=500,  # Adjust chunk size
    chunk_overlap=50  # Overlap between chunks
)

# Upload single file
result = uploader.upload_file(
    file_path="/path/to/document.pdf",
    category="Manual",
    metadata={
        "author": "IT Department",
        "version": "1.0",
        "date": "2026-01-21"
    }
)

# Upload directory
results = uploader.upload_directory(
    directory_path="/path/to/documents",
    category="Documentation",
    recursive=True
)

# Get statistics
stats = uploader.get_collection_stats()
print(stats)
```

## 📂 Method 4: Copy Files to Container

If you have many files on your host machine:

```bash
# Create a documents directory in the container
docker exec -it agent2_ticket mkdir -p /app/documents

# Copy files to container
docker cp /path/to/local/documents/. agent2_ticket:/app/documents/

# Upload from inside container
docker exec -it agent2_ticket python upload_documents.py \
  /app/documents \
  --category "FAQ" \
  --recursive \
  --stats
```

## 🔧 Docker Volume Method (Best for Production)

Update `docker-compose.yml` to mount a documents folder:

```yaml
services:
  agent2_ticket:
    volumes:
      - ./documents:/app/documents:ro  # Read-only mount
    # ... rest of config
```

Then:

```bash
# Place your documents in the folder
cp /path/to/docs/* /Users/karolsliwka/Desktop/ai_stack/documents/

# Restart container
docker-compose restart agent2_ticket

# Upload from container
docker exec -it agent2_ticket python upload_documents.py \
  /app/documents \
  --category "Documentation" \
  --recursive
```

## 📋 Example: Complete Workflow

### 1. Prepare Your Documents

```bash
# Create a directory with your documents
mkdir -p ~/ticket_docs
cp important_faq.pdf ~/ticket_docs/
cp user_manual.docx ~/ticket_docs/
cp policies.pdf ~/ticket_docs/
```

### 2. Upload via API

```bash
# Upload FAQ
curl -X POST http://localhost:8002/upload \
  -F "file=@$HOME/ticket_docs/important_faq.pdf" \
  -F "category=FAQ"

# Upload Manual
curl -X POST http://localhost:8002/upload \
  -F "file=@$HOME/ticket_docs/user_manual.docx" \
  -F "category=Manual"

# Upload Policy
curl -X POST http://localhost:8002/upload \
  -F "file=@$HOME/ticket_docs/policies.pdf" \
  -F "category=Policy"
```

### 3. Verify Upload

```bash
# Check uploaded documents
curl http://localhost:8002/documents | jq

# Check statistics
curl http://localhost:8002/stats | jq
```

### 4. Test RAG Search

```bash
# Test if uploaded docs are searchable
curl -X POST http://localhost:8002/run \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Jak mogę zmienić hasło?",
    "step": "initial"
  }' | jq
```

## 🎯 Categories for Different Document Types

Suggested category names:

- **FAQ** - Frequently asked questions
- **Manual** - User manuals and guides
- **Policy** - University policies
- **Procedure** - Step-by-step procedures
- **Form** - Forms and templates
- **Contact** - Contact information
- **Schedule** - Schedules and calendars
- **IT** - IT documentation
- **Academic** - Academic information
- **Administrative** - Administrative documents

## 📊 How Text Extraction Works

### PDF Files
- Uses `pdfplumber` (primary) or `PyPDF2` (fallback)
- Extracts text from all pages
- Handles multi-column layouts

### Word Documents (.docx)
- Extracts all paragraphs
- Extracts text from tables
- Preserves formatting where possible

### Excel Files (.xlsx)
- Processes all sheets
- Converts tables to readable text
- Includes sheet names

### Text Files (.txt, .md)
- Direct text extraction
- UTF-8 encoding

### JSON Files
- Pretty-prints JSON structure
- Converts to readable format

## 🧩 How Chunking Works

Documents are split into chunks for better RAG performance:

- **Default chunk size**: 500 words
- **Overlap**: 50 words (ensures context continuity)
- Each chunk is embedded separately
- Search returns most relevant chunks

## 🔍 Testing Your Uploads

After uploading documents, test with queries:

```bash
# Test 1: Simple query
curl -X POST http://localhost:8002/run \
  -H "Content-Type: application/json" \
  -d '{"input": "Jak się zarejestrować?", "step": "initial"}'

# Test 2: Check RAG results
curl -X POST http://localhost:8002/run \
  -H "Content-Type: application/json" \
  -d '{"input": "Gdzie znaleźć formularze?", "step": "initial"}' \
  | jq '.rag_results'

# Test 3: Category-specific query
curl -X POST http://localhost:8002/run \
  -H "Content-Type: application/json" \
  -d '{"input": "Jaka jest polityka prywatności?", "step": "initial"}'
```

## 🧹 Maintenance

### Delete Outdated Documents

```bash
# List all documents
curl http://localhost:8002/documents

# Delete specific document
curl -X DELETE http://localhost:8002/documents/old_manual.pdf
```

### Re-upload Updated Documents

```bash
# Delete old version
curl -X DELETE http://localhost:8002/documents/manual.pdf

# Upload new version
curl -X POST http://localhost:8002/upload \
  -F "file=@manual_v2.pdf" \
  -F "category=Manual"
```

## ⚠️ Troubleshooting

### "Unsupported file format"
- Check that your file extension is supported
- Supported: `.pdf`, `.docx`, `.doc`, `.xlsx`, `.xls`, `.txt`, `.md`, `.json`

### "No text content found"
- File might be an image-based PDF (requires OCR)
- File might be corrupted
- Try opening the file locally first

### "Connection refused"
- Ensure Ollama is running and healthy
- Check that llama3 model is pulled: `docker exec ollama ollama list`
- Verify Qdrant is running: `curl http://localhost:6333/healthz`

### Slow upload
- Large files take time to process
- Each chunk needs to be embedded by Ollama
- Consider breaking large files into smaller ones

## 📚 Related Documentation

- [Agent 2 API Documentation](http://localhost:8002/docs)
- [Qdrant Dashboard](http://localhost:6333/dashboard)
- [Main README](../../README.md)
- [Node-RED Payloads](../../nodered/PAYLOADS.md)
