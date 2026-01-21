#!/bin/bash

echo "🚀 Starting AI Stack..."
echo ""

# Check if docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Build and start all services
echo "📦 Building and starting all services..."
docker-compose up -d --build

echo ""
echo "⏳ Waiting for services to be healthy..."
echo ""

# Wait for services to be healthy
sleep 10

# Check Ollama
echo "🔍 Checking Ollama..."
for i in {1..30}; do
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo "✅ Ollama is ready"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  Ollama might still be starting up"
    fi
    sleep 2
done

# Check Qdrant
echo "🔍 Checking Qdrant..."
if curl -s http://localhost:6333/healthz > /dev/null 2>&1; then
    echo "✅ Qdrant is ready"
else
    echo "⚠️  Qdrant might still be starting up"
fi

# Pull llama3 model if not present
echo ""
echo "📥 Checking if llama3 model is available..."
if ! docker exec ollama ollama list | grep -q "llama3"; then
    echo "📥 Pulling llama3 model (this may take a few minutes)..."
    docker exec ollama ollama pull llama3
    echo "✅ llama3 model ready"
else
    echo "✅ llama3 model already present"
fi

echo ""
echo "🎉 AI Stack is ready!"
echo ""
echo "📊 Service URLs:"
echo "   - Qdrant Dashboard: http://localhost:6333/dashboard"
echo "   - Open WebUI: http://localhost:3000"
echo "   - Node-RED: http://localhost:1880"
echo "   - Agent 1 (Student): http://localhost:8001"
echo "   - Agent 2 (Ticket): http://localhost:8002"
echo "   - Agent 3 (Analytics): http://localhost:8003"
echo "   - Agent 4 (BOS): http://localhost:8004"
echo "   - Agent 5 (Security): http://localhost:8005"
echo ""
echo "📝 View logs with: docker-compose logs -f"
echo "🛑 Stop services with: docker-compose down"
echo ""
