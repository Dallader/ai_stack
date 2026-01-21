#!/bin/bash

echo "🧪 Testing AI Stack Services..."
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

test_service() {
    local name=$1
    local url=$2
    
    if curl -s -f "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ $name is responding${NC}"
        return 0
    else
        echo -e "${RED}❌ $name is not responding${NC}"
        return 1
    fi
}

test_api() {
    local name=$1
    local url=$2
    local data=$3
    
    response=$(curl -s -X POST "$url" \
        -H "Content-Type: application/json" \
        -d "$data" 2>&1)
    
    if [ $? -eq 0 ] && [[ ! "$response" =~ "error" ]]; then
        echo -e "${GREEN}✅ $name API is working${NC}"
        return 0
    else
        echo -e "${RED}❌ $name API failed${NC}"
        echo -e "${YELLOW}   Response: ${response:0:100}${NC}"
        return 1
    fi
}

echo "📋 Testing Infrastructure Services..."
test_service "Qdrant" "http://localhost:6333/healthz"
test_service "Ollama" "http://localhost:11434/api/tags"
echo ""

echo "📋 Testing Application Services..."
test_service "Open WebUI" "http://localhost:3000"
test_service "Node-RED" "http://localhost:1880"
echo ""

echo "📋 Testing Agent Health Endpoints..."
test_service "Agent 1 (Student)" "http://localhost:8001/health"
test_service "Agent 2 (Ticket)" "http://localhost:8002/health"
test_service "Agent 3 (Analytics)" "http://localhost:8003/health"
test_service "Agent 4 (BOS)" "http://localhost:8004/health"
test_service "Agent 5 (Security)" "http://localhost:8005/health"
echo ""

echo "📋 Testing Agent APIs..."
test_api "Agent 1" "http://localhost:8001/run" '{"input": "Hello, test query"}'
test_api "Agent 2" "http://localhost:8002/run" '{"input": "Test ticket", "step": "initial"}'
test_api "Agent 3" "http://localhost:8003/run" '{"input": "Test data"}'
echo ""

echo "📋 Checking Docker Containers..."
echo ""
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "(qdrant|ollama|open-webui|agent|node-red)"
echo ""

echo "📋 Checking Qdrant Collections..."
collections=$(curl -s http://localhost:6333/collections | grep -o '"name":"[^"]*"' | cut -d'"' -f4)
if [ ! -z "$collections" ]; then
    echo -e "${GREEN}Collections found:${NC}"
    echo "$collections" | while read col; do
        echo "  - $col"
    done
else
    echo -e "${YELLOW}⚠️  No collections found yet${NC}"
fi
echo ""

echo "📋 Checking Ollama Models..."
models=$(docker exec ollama ollama list 2>/dev/null | tail -n +2)
if [ ! -z "$models" ]; then
    echo -e "${GREEN}Models found:${NC}"
    echo "$models"
else
    echo -e "${YELLOW}⚠️  No models found. Run: docker exec ollama ollama pull llama3${NC}"
fi
echo ""

echo "🎉 Test complete!"
echo ""
echo "💡 Tips:"
echo "   - Access Qdrant Dashboard: http://localhost:6333/dashboard"
echo "   - Access Open WebUI: http://localhost:3000"
echo "   - View logs: docker-compose logs -f"
echo ""
