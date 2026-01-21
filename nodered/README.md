# Node-RED Flows for AI Stack

This directory contains Node-RED flow configurations for orchestrating the AI agents.

## 📋 Available Flows

### 1. Agent 2 - Ticket System Flow (`agent2_ticket_flow.json`)

Complete workflow for the ticket system with multi-step conversation handling.

**Features:**
- Initial query with RAG search
- Automatic handling of "need_details" responses
- Ticket creation with categorization
- BOS notification on ticket creation

**Example Payloads:**

```json
// Initial Query
{
  "input": "Jak mogę zmienić hasło?",
  "step": "initial"
}

// Query with Additional Details
{
  "input": "Nie mogę się zalogować do systemu",
  "step": "need_details",
  "original_query": "Jak mogę zmienić hasło?"
}

// Direct Ticket Creation
{
  "input": "Mój laptop nie działa",
  "step": "create_ticket",
  "additional_info": "Laptop nie włącza się od wczoraj"
}
```

### 2. All Agents Test Dashboard (`all_agents_flow.json`)

Test interface for all 5 agents with health checks.

**Example Payloads:**

```json
// Agent 1 - Student
{
  "input": "Explain machine learning in simple terms"
}

// Agent 2 - Ticket
{
  "input": "Jak mogę zmienić hasło?",
  "step": "initial"
}

// Agent 3 - Analytics
{
  "input": "Sales data: Q1: 100k, Q2: 150k, Q3: 200k, Q4: 180k. Analyze trends."
}

// Agent 4 - BOS
{
  "input": "Process new ticket notification"
}

// Agent 5 - Security
{
  "input": "Audit user login attempt from IP 192.168.1.100"
}
```

### 3. Multi-Agent Orchestration (`orchestration_flow.json`)

Complete workflow that routes queries to appropriate agents and handles multi-agent coordination.

**Flow:**
1. User query → Security audit log
2. Query routing (based on keywords)
3. Agent processing
4. Response handling (completed/ticket/need_details)
5. BOS notification (if ticket created)
6. Analytics logging
7. User notification

**Example Payload:**

```json
{
  "payload": "Mam problem z dostępem do systemu ocen",
  "user": {
    "id": "user123",
    "email": "student@university.edu",
    "name": "Jan Kowalski"
  }
}
```

## 🚀 How to Use

### Import Flows into Node-RED

1. Start your AI stack:
   ```bash
   cd /Users/karolsliwka/Desktop/ai_stack
   ./start.sh
   ```

2. Open Node-RED:
   ```
   http://localhost:1880
   ```

3. Import a flow:
   - Click the menu (☰) → Import
   - Click "select a file to import"
   - Choose one of the JSON files from `flows/` directory
   - Click "Import"

4. Deploy:
   - Click the "Deploy" button (top right)

### Test the Flows

**Method 1: Using Inject Nodes**
- Click the blue button on any "inject" node to trigger the flow
- View results in the Debug panel (right sidebar)

**Method 2: Using curl**

```bash
# Test via Node-RED HTTP endpoint (if configured)
curl -X POST http://localhost:1880/agent2 \
  -H "Content-Type: application/json" \
  -d '{"input": "Test query", "step": "initial"}'
```

**Method 3: Direct Agent API**

```bash
# Agent 2 - Initial Query
curl -X POST http://localhost:8002/run \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Jak mogę zmienić hasło?",
    "step": "initial"
  }'

# Agent 2 - With Details
curl -X POST http://localhost:8002/run \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Nie mogę się zalogować",
    "step": "need_details",
    "original_query": "Jak mogę zmienić hasło?"
  }'

# Agent 2 - Create Ticket
curl -X POST http://localhost:8002/run \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Laptop nie działa",
    "step": "create_ticket",
    "additional_info": "Nie włącza się"
  }'
```

## 📊 Payload Structure Reference

### Agent 1 (Student)
```json
{
  "input": "Your question here"
}
```

### Agent 2 (Ticket)
```json
{
  "input": "User query",
  "step": "initial|need_details|create_ticket",
  "original_query": "Original query text (for need_details step)",
  "additional_info": "Additional information (optional)"
}
```

### Agent 3 (Analytics)
```json
{
  "input": "Data to analyze"
}
```

### Agent 4 (BOS)
```json
{
  "input": "BOS notification or request"
}
```

### Agent 5 (Security)
```json
{
  "input": "Security audit log or event"
}
```

## 🔍 Response Structure

### Agent 2 Responses

**Step: completed (RAG found results)**
```json
{
  "response": "Answer based on RAG documents",
  "rag_results": [
    {
      "text": "Document content",
      "category": "Category name",
      "score": 0.85
    }
  ],
  "step": "completed",
  "collection": "agent2_tickets"
}
```

**Step: need_details (No RAG results)**
```json
{
  "response": "Nie znalazłem odpowiedzi. Czy możesz podać więcej szczegółów?",
  "step": "need_details",
  "original_query": "Original user query",
  "collection": "agent2_tickets"
}
```

**Step: ticket_created**
```json
{
  "response": "Ticket creation confirmation message",
  "ticket": {
    "query": "User query",
    "additional_info": "Additional details",
    "category": "Assigned category",
    "priority": "Priority level",
    "status": "created"
  },
  "step": "ticket_created",
  "collection": "agent2_tickets"
}
```

## 🛠️ Customization

### Add HTTP Endpoints

To expose flows via HTTP:

1. Add an "http in" node at the start
2. Add an "http response" node at the end
3. Configure the endpoint path

Example:
```
[http in: POST /api/ticket] → [function] → [http request] → [http response]
```

### Add Email Notifications

1. Install email node: `Manage palette → Install → node-red-node-email`
2. Configure SMTP settings
3. Add email node after ticket creation

### Add Database Storage

1. Install database nodes (PostgreSQL, MongoDB, etc.)
2. Store tickets/responses in database
3. Query historical data

## 📝 Tips

- Use Debug nodes liberally to see message flow
- Use Function nodes to transform payloads
- Use Switch nodes for conditional routing
- Store data in flow/global context for multi-step conversations
- Add delays between API calls to avoid overload

## 🔗 Related Documentation

- [Main README](../README.md)
- [Docker Commands](../COMMANDS.md)
- [Agent 2 app.py](../agents/agent2_ticket/app.py)

## 🤝 Contributing

To add new flows:
1. Create the flow in Node-RED
2. Export: Menu → Export → JSON
3. Save to `flows/` directory with descriptive name
4. Update this README with payload examples
