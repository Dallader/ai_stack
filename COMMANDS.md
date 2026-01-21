# Quick Reference - Docker Compose Commands

## 🚀 Starting Services

```bash
# Start all services (recommended)
./start.sh

# Or manually with docker-compose
docker-compose up -d --build

# Start specific service
docker-compose up -d agent2_ticket

# Start without building
docker-compose up -d
```

## 🛑 Stopping Services

```bash
# Stop all services (recommended)
./stop.sh

# Or manually
docker-compose down

# Stop but keep volumes (data preserved)
docker-compose stop

# Stop and remove volumes (⚠️ deletes all data)
docker-compose down -v
```

## 🔍 Monitoring & Debugging

```bash
# View all logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f agent2_ticket
docker-compose logs -f ollama
docker-compose logs -f qdrant

# View last 100 lines
docker-compose logs --tail=100 agent2_ticket

# Check service status
docker-compose ps

# Check resource usage
docker stats
```

## 🔄 Restarting Services

```bash
# Restart all services
docker-compose restart

# Restart specific service
docker-compose restart agent2_ticket

# Rebuild and restart specific service
docker-compose up -d --build agent2_ticket

# Force recreate containers
docker-compose up -d --force-recreate
```

## 🧪 Testing

```bash
# Run test script
./test.sh

# Test specific agent health
curl http://localhost:8001/health
curl http://localhost:8002/health

# Test agent API
curl -X POST http://localhost:8002/run \
  -H "Content-Type: application/json" \
  -d '{"input": "Test", "step": "initial"}'

# Check Qdrant
curl http://localhost:6333/healthz
curl http://localhost:6333/collections

# Check Ollama
curl http://localhost:11434/api/tags
docker exec ollama ollama list
```

## 🔧 Maintenance

```bash
# Pull latest images
docker-compose pull

# Rebuild all services
docker-compose build

# Rebuild specific service
docker-compose build agent2_ticket

# Remove unused containers/images
docker system prune

# Remove unused volumes
docker volume prune

# View volumes
docker volume ls

# Inspect volume
docker volume inspect ai_stack_qdrant_data
```

## 📊 Ollama Management

```bash
# Pull llama3 model
docker exec ollama ollama pull llama3

# List models
docker exec ollama ollama list

# Remove model
docker exec ollama ollama rm llama3

# Run model interactively
docker exec -it ollama ollama run llama3

# Check model info
docker exec ollama ollama show llama3
```

## 🗄️ Qdrant Management

```bash
# List collections
curl http://localhost:6333/collections

# Get collection info
curl http://localhost:6333/collections/agent2_tickets

# Delete collection
curl -X DELETE http://localhost:6333/collections/agent2_tickets

# Count points in collection
curl http://localhost:6333/collections/agent2_tickets/points/count

# Access Qdrant Dashboard
open http://localhost:6333/dashboard
```

## 🔐 Accessing Container Shell

```bash
# Access agent container
docker exec -it agent2_ticket /bin/bash

# Access Ollama container
docker exec -it ollama /bin/bash

# Access Qdrant container
docker exec -it qdrant /bin/sh

# Run Python commands in agent
docker exec -it agent2_ticket python -c "import qdrant_client; print('OK')"
```

## 📦 Backup & Restore

```bash
# Backup Qdrant data
docker run --rm -v ai_stack_qdrant_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/qdrant_backup.tar.gz /data

# Restore Qdrant data
docker run --rm -v ai_stack_qdrant_data:/data -v $(pwd):/backup \
  alpine tar xzf /backup/qdrant_backup.tar.gz -C /

# Backup Ollama models
docker run --rm -v ai_stack_ollama_data:/data -v $(pwd):/backup \
  alpine tar czf /backup/ollama_backup.tar.gz /data

# Export logs
docker-compose logs > ai_stack_logs.txt
```

## 🌐 Network Management

```bash
# List networks
docker network ls

# Inspect ai_network
docker network inspect ai_stack_ai_network

# Test network connectivity
docker exec agent2_ticket ping qdrant
docker exec agent2_ticket ping ollama
docker exec agent2_ticket curl http://qdrant:6333/healthz
```

## 🔥 Troubleshooting

```bash
# Complete reset (⚠️ deletes everything)
docker-compose down -v
docker system prune -a --volumes
./start.sh

# Restart specific service with fresh build
docker-compose stop agent2_ticket
docker-compose rm -f agent2_ticket
docker-compose build agent2_ticket
docker-compose up -d agent2_ticket

# Check health status
docker inspect agent2_ticket | grep -A 10 Health

# View service dependencies
docker-compose config | grep depends_on -A 5
```

## 📱 Service URLs

- **Qdrant Dashboard**: http://localhost:6333/dashboard
- **Open WebUI**: http://localhost:3000
- **Node-RED**: http://localhost:1880
- **Agent 1 API Docs**: http://localhost:8001/docs
- **Agent 2 API Docs**: http://localhost:8002/docs
- **Agent 3 API Docs**: http://localhost:8003/docs
- **Agent 4 API Docs**: http://localhost:8004/docs
- **Agent 5 API Docs**: http://localhost:8005/docs

## 💡 Common Issues

### "Port already in use"
```bash
# Find process using port
lsof -i :8002

# Kill process
kill -9 <PID>

# Or change port in docker-compose.yml
```

### "Cannot connect to Docker daemon"
```bash
# Start Docker Desktop
open -a Docker

# Or restart Docker
killall Docker && open -a Docker
```

### "Model not found"
```bash
# Pull llama3 model
docker exec ollama ollama pull llama3
```

### "Collection not found"
```bash
# Restart agent to reinitialize
docker-compose restart agent2_ticket
```
