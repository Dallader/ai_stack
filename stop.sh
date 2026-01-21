#!/bin/bash

echo "🛑 Stopping AI Stack..."
docker-compose down

echo ""
echo "✅ All services stopped"
echo ""
echo "💡 To remove all data, use: docker-compose down -v"
echo ""
