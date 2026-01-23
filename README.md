# AI Stack - Multi-Agent System

A complete AI stack with 5 specialized agents, orchestration via Node-RED, vector database (Qdrant), and LLM support (Ollama).

## 🏗️ Architecture

```
├── Infrastructure Layer
│   ├── Qdrant (Vector Database) - Port 6333
│   └── Ollama (LLM Server) - Port 11434
├── Application Layer  
│   ├── Open WebUI - Port 3000
│   ├── Agent 1 (Student) - Port 8001
│   ├── Agent 2 (Ticket) - Port 8002
│   ├── Agent 3 (Analytics) - Port 8003
│   ├── Agent 4 (BOS) - Port 8004
│   └── Agent 5 (Security) - Port 8005
└── Orchestration Layer
    └── Node-RED - Port 1880
```

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed and running
- At least 8GB RAM available
- 20GB free disk space

### Start Services

The startup behavior depends on the `start_necessary` setting in `agents/agent2_ticket/host_settings.json`:

- If `start_necessary: true` → Starts only essential services: Qdrant, Ollama, Agent2 (Ticket), Node-RED
- If `start_necessary: false` → Starts all services

```bash
# Make the script executable
chmod +x start.py

# Start the stack (automatically checks host_settings.json)
python3 start.py
```

The script will:
1. ✅ Check Docker is running
2. 📦 Build selected services
3. 🔍 Wait for services to be healthy
4. 📥 Pull the required models if needed

### Manual Commands

```bash
# Start all services (ignores host_settings.json)
docker-compose up -d --build

# Start only necessary services
docker-compose --profile necessary up -d --build

# View logs
docker-compose logs -f

# View logs for specific service
docker-compose logs -f agent2_ticket

# Stop all services
docker-compose down

# Stop and remove all data
docker-compose down -v

# Restart a specific service
docker-compose restart agent2_ticket

# Rebuild a specific service
docker-compose up -d --build agent2_ticket
```

## 📊 Service Details

### Infrastructure Services

**Qdrant** (Vector Database)
- Dashboard: http://localhost:6333/dashboard
- API: http://localhost:6333
- Storage: Persistent volume `qdrant_data`

**Ollama** (LLM Server)
- API: http://localhost:11434
- Model: llama3
- Storage: Persistent volume `ollama_data`

### Application Services

**Open WebUI**
- URL: http://localhost:3000
- Chat interface for Ollama models
- Storage: Persistent volume `open_webui_data`

**Agent 1 - Student Agent**
- URL: http://localhost:8001
- Collection: `agent1_student`
- Purpose: Student support and queries

**Agent 2 - Ticket Agent**
- URL: http://localhost:8002
- Collection: `agent2_tickets`
- Purpose: Ticket management with RAG
- Features:
  - RAG-based document search
  - Automatic ticket categorization
  - Priority assignment
  - Multi-step conversation flow

**Agent 3 - Analytics Agent**
- URL: http://localhost:8003
- Collection: `agent3_stats`
- Purpose: Data analysis and insights

**Agent 4 - BOS Agent**
- URL: http://localhost:8004
- Collection: `agent4_bos`
- Purpose: Business operations support

**Agent 5 - Security Agent**
- URL: http://localhost:8005
- Collection: `agent5_audit`
- Purpose: Security and audit logging

### Orchestration

**Node-RED**
- URL: http://localhost:1880
- Visual workflow orchestration
- Storage: Persistent volume `nodered_data`

## 🔧 Testing Agents

### Test Agent 2 (Ticket System)

```bash
# Initial query - will search RAG database
curl -X POST http://localhost:8002/run \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Jak mogę zmienić hasło?",
    "step": "initial"
  }'

# Provide additional details
curl -X POST http://localhost:8002/run \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Nie mogę się zalogować do systemu",
    "step": "need_details",
    "original_query": "Jak mogę zmienić hasło?"
  }'

# Direct ticket creation
curl -X POST http://localhost:8002/run \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Mój laptop nie działa",
    "step": "create_ticket",
    "additional_info": "Laptop nie włącza się"
  }'
```

### Test Other Agents

```bash
# Agent 1 - Student
curl -X POST http://localhost:8001/run \
  -H "Content-Type: application/json" \
  -d '{"input": "What is machine learning?"}'

# Agent 3 - Analytics
curl -X POST http://localhost:8003/run \
  -H "Content-Type: application/json" \
  -d '{"input": "Sales data: Q1: 100k, Q2: 150k, Q3: 200k"}'

# Health checks
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

## 🐛 Troubleshooting

### Services not starting

```bash
# Check logs
docker-compose logs

# Check specific service
docker-compose logs qdrant
docker-compose logs ollama
```

### Ollama model not found

```bash
# Pull llama3 model manually
docker exec -it ollama ollama pull llama3

# List available models
docker exec -it ollama ollama list
```

### Qdrant connection issues

```bash
# Check Qdrant is healthy
curl http://localhost:6333/healthz

# Check collections
curl http://localhost:6333/collections
```

### Agent errors

```bash
# Restart specific agent
docker-compose restart agent2_ticket

# Rebuild agent
docker-compose up -d --build agent2_ticket

# Check agent logs
docker-compose logs -f agent2_ticket
```

### Reset everything

```bash
# Stop and remove all containers, networks, volumes
docker-compose down -v

# Start fresh
./start.sh
```

## 📝 Configuration

### Environment Variables

Edit `docker-compose.yml` to modify:
- `COLLECTION`: Qdrant collection name for each agent
- `QDRANT_URL`: Qdrant connection URL
- `OLLAMA_BASE_URL`: Ollama API URL

### Agent Customization

Each agent has its own directory:
- `app.py`: Main application code
- `requirements.txt`: Python dependencies
- `Dockerfile`: Container build instructions
- `json/`: Configuration files (Agent 2 only)

## 🔄 Startup Order

The docker-compose file ensures proper startup order:

1. **Infrastructure** (Qdrant, Ollama) - start first with health checks
2. **Open WebUI** - starts after Ollama is healthy
3. **All Agents** - start after both Qdrant and Ollama are healthy
4. **Node-RED** - starts last after all agents are running

## 📚 API Documentation

Once services are running, access:
- Agent 1 Docs: http://localhost:8001/docs
- Agent 2 Docs: http://localhost:8002/docs
- Agent 3 Docs: http://localhost:8003/docs
- Agent 4 Docs: http://localhost:8004/docs
- Agent 5 Docs: http://localhost:8005/docs

## 💾 Data Persistence

All data is stored in Docker volumes:
- `qdrant_data`: Vector embeddings and collections
- `ollama_data`: LLM models
- `open_webui_data`: Chat history and settings
- `nodered_data`: Node-RED flows and configuration

To backup data:
```bash
docker-compose down
docker run --rm -v ai_stack_qdrant_data:/data -v $(pwd):/backup alpine tar czf /backup/qdrant_backup.tar.gz /data
```

## 🤝 Contributing

1. Make changes to agent code
2. Test locally: `docker-compose up -d --build agent_name`
3. Commit changes
4. Push to repository

## 📄 License

[Your License Here]
