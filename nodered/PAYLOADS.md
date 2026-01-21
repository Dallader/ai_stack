# Node-RED Message Payloads Reference

Quick reference guide for testing all agents via Node-RED or curl.

## 🎯 Agent 1 - Student Support

### Simple Query
```json
{
  "input": "What is machine learning?"
}
```

### Course Question
```json
{
  "input": "Explain the difference between supervised and unsupervised learning"
}
```

### Homework Help
```json
{
  "input": "How do I solve quadratic equations?"
}
```

**Test in Node-RED:**
```javascript
msg.payload = {
    "input": "What is machine learning?"
};
msg.url = "http://agent1_student:8000/run";
return msg;
```

**Test with curl:**
```bash
curl -X POST http://localhost:8001/run \
  -H "Content-Type: application/json" \
  -d '{"input": "What is machine learning?"}'
```

---

## 🎫 Agent 2 - Ticket System

### Step 1: Initial Query (RAG Search)
```json
{
  "input": "Jak mogę zmienić hasło?",
  "step": "initial"
}
```

**Expected Responses:**
- If found in RAG: `{"step": "completed", "response": "...", "rag_results": [...]}`
- If not found: `{"step": "need_details", "original_query": "..."}`

### Step 2: Provide Additional Details
```json
{
  "input": "Nie mogę się zalogować do systemu",
  "step": "need_details",
  "original_query": "Jak mogę zmienić hasło?"
}
```

**Expected Responses:**
- If found in RAG: `{"step": "completed", "response": "...", "rag_results": [...]}`
- If still not found: `{"step": "ticket_created", "ticket": {...}}`

### Step 3: Direct Ticket Creation
```json
{
  "input": "Mój laptop nie działa",
  "step": "create_ticket",
  "additional_info": "Laptop nie włącza się od wczoraj"
}
```

**Response:**
```json
{
  "response": "Zgłoszenie utworzone...",
  "ticket": {
    "query": "Mój laptop nie działa",
    "additional_info": "Laptop nie włącza się od wczoraj",
    "category": "Sprzęt",
    "priority": "Wysoki",
    "status": "created"
  },
  "step": "ticket_created"
}
```

### More Examples

**Password Reset:**
```json
{
  "input": "Jak zresetować hasło do systemu?",
  "step": "initial"
}
```

**System Access:**
```json
{
  "input": "Nie mam dostępu do biblioteki online",
  "step": "initial"
}
```

**Technical Issue:**
```json
{
  "input": "Projektor w sali 201 nie działa",
  "step": "create_ticket",
  "additional_info": "Projektor nie wyświetla obrazu, lampka mruga na czerwono"
}
```

**Test in Node-RED:**
```javascript
msg.payload = {
    "input": "Jak mogę zmienić hasło?",
    "step": "initial"
};
msg.url = "http://agent2_ticket:8000/run";
return msg;
```

**Test with curl:**
```bash
# Initial query
curl -X POST http://localhost:8002/run \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Jak mogę zmienić hasło?",
    "step": "initial"
  }'

# With additional details
curl -X POST http://localhost:8002/run \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Nie mogę się zalogować",
    "step": "need_details",
    "original_query": "Jak mogę zmienić hasło?"
  }'

# Create ticket directly
curl -X POST http://localhost:8002/run \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Laptop nie działa",
    "step": "create_ticket",
    "additional_info": "Nie włącza się"
  }'
```

---

## 📊 Agent 3 - Analytics

### Sales Data Analysis
```json
{
  "input": "Sales data: Q1: 100k, Q2: 150k, Q3: 200k, Q4: 180k. Analyze trends and provide insights."
}
```

### Student Performance
```json
{
  "input": "Student scores: Math: 85, Physics: 92, Chemistry: 78, Biology: 88. Analyze performance."
}
```

### Ticket Statistics
```json
{
  "input": "Ticket stats: Open: 45, In Progress: 23, Resolved: 312, Closed: 298. What's the trend?"
}
```

### Website Traffic
```json
{
  "input": "Website traffic: Monday: 1200, Tuesday: 1450, Wednesday: 1380, Thursday: 1520, Friday: 1890. Analyze patterns."
}
```

**Test in Node-RED:**
```javascript
msg.payload = {
    "input": "Sales data: Q1: 100k, Q2: 150k, Q3: 200k, Q4: 180k. Analyze trends."
};
msg.url = "http://agent3_analytics:8000/run";
return msg;
```

**Test with curl:**
```bash
curl -X POST http://localhost:8003/run \
  -H "Content-Type: application/json" \
  -d '{"input": "Sales data: Q1: 100k, Q2: 150k, Q3: 200k. Analyze."}'
```

---

## 🏢 Agent 4 - BOS (Business Operations)

### Ticket Notification
```json
{
  "input": "New ticket created: Category=Sprzęt, Priority=Wysoki, Query=Laptop nie działa"
}
```

### Status Update
```json
{
  "input": "Update ticket status: Ticket #12345 resolved"
}
```

### Department Query
```json
{
  "input": "Process department request for new equipment"
}
```

**Test in Node-RED:**
```javascript
msg.payload = {
    "input": "New ticket: Category=IT, Priority=High"
};
msg.url = "http://agent4_bos:8000/run";
return msg;
```

**Test with curl:**
```bash
curl -X POST http://localhost:8004/run \
  -H "Content-Type: application/json" \
  -d '{"input": "New ticket notification"}'
```

---

## 🔐 Agent 5 - Security & Audit

### Login Audit
```json
{
  "input": "Audit user login attempt from IP 192.168.1.100"
}
```

### Security Event
```json
{
  "input": "Log security event: Multiple failed login attempts detected for user: john.doe"
}
```

### Access Request
```json
{
  "input": "User requested access to admin panel: user_id=12345, timestamp=2026-01-21T17:00:00Z"
}
```

### System Change
```json
{
  "input": "System configuration changed: Database connection pool size increased from 10 to 20"
}
```

**Test in Node-RED:**
```javascript
msg.payload = {
    "input": "Audit login from IP 192.168.1.100"
};
msg.url = "http://agent5_security:8000/run";
return msg;
```

**Test with curl:**
```bash
curl -X POST http://localhost:8005/run \
  -H "Content-Type: application/json" \
  -d '{"input": "Audit user login from IP 192.168.1.100"}'
```

---

## 🔄 Multi-Agent Orchestration Payload

Complete workflow with user context:

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

**Flow:**
1. Security logs the query
2. Router determines target agent
3. Agent processes request
4. If ticket created → notify BOS
5. Log to analytics
6. Send response to user

**Test in Node-RED:**
```javascript
msg.payload = "Mam problem z dostępem do systemu";
msg.user = {
    "id": "user123",
    "email": "student@university.edu",
    "name": "Jan Kowalski"
};
return msg;
```

---

## 💡 Node-RED Function Node Examples

### 1. Basic HTTP Request
```javascript
msg.payload = {
    "input": "Your query here"
};
msg.url = "http://agent2_ticket:8000/run";
msg.headers = {
    "Content-Type": "application/json"
};
return msg;
```

### 2. Handle Response
```javascript
// Check if response is successful
if (msg.payload.step === "completed") {
    msg.status = "success";
    msg.response = msg.payload.response;
} else if (msg.payload.step === "need_details") {
    msg.status = "awaiting_details";
    msg.original_query = msg.payload.original_query;
} else if (msg.payload.step === "ticket_created") {
    msg.status = "ticket_created";
    msg.ticket = msg.payload.ticket;
}
return msg;
```

### 3. Retry Logic
```javascript
// Store retry count in flow context
let retryCount = flow.get('retryCount') || 0;

if (retryCount < 3) {
    // Prepare retry payload
    msg.payload = {
        "input": flow.get('original_query'),
        "step": "need_details",
        "original_query": flow.get('original_query')
    };
    flow.set('retryCount', retryCount + 1);
    return msg;
} else {
    // Max retries reached, create ticket
    msg.payload = {
        "input": flow.get('original_query'),
        "step": "create_ticket"
    };
    flow.set('retryCount', 0);
    return msg;
}
```

### 4. Multi-Agent Coordinator
```javascript
// Route based on keywords
const query = msg.payload.toLowerCase();
let targetAgent = "";
let targetUrl = "";

if (query.includes("analiz") || query.includes("statyst")) {
    targetAgent = "agent3_analytics";
    targetUrl = "http://agent3_analytics:8000/run";
} else if (query.includes("pytanie") || query.includes("nauk")) {
    targetAgent = "agent1_student";
    targetUrl = "http://agent1_student:8000/run";
} else {
    targetAgent = "agent2_ticket";
    targetUrl = "http://agent2_ticket:8000/run";
    msg.payload = {
        "input": msg.payload,
        "step": "initial"
    };
}

msg.url = targetUrl;
msg.targetAgent = targetAgent;
return msg;
```

### 5. Error Handler
```javascript
if (msg.statusCode !== 200) {
    msg.error = {
        "message": "Agent request failed",
        "statusCode": msg.statusCode,
        "agent": msg.targetAgent,
        "timestamp": new Date().toISOString()
    };
    // Send to error logging agent
    msg.url = "http://agent5_security:8000/run";
    msg.payload = {
        "input": `Error: ${msg.error.message} for agent ${msg.targetAgent}`
    };
}
return msg;
```

---

## 🧪 Complete Test Scenarios

### Scenario 1: Successful RAG Query
```javascript
// Step 1: Send initial query
msg.payload = {
    "input": "Jak mogę zmienić hasło?",
    "step": "initial"
};
// Expected: RAG finds answer, step="completed"
```

### Scenario 2: RAG Miss → Ticket Created
```javascript
// Step 1: Initial query (no RAG match)
msg.payload = {
    "input": "Mój nietypowy problem",
    "step": "initial"
};
// Expected: step="need_details"

// Step 2: Provide details (still no match)
msg.payload = {
    "input": "Szczegółowy opis problemu",
    "step": "need_details",
    "original_query": "Mój nietypowy problem"
};
// Expected: step="ticket_created"
```

### Scenario 3: Multi-Agent Workflow
```javascript
// User query triggers multiple agents
msg.payload = "Mam problem z dostępem";
msg.user = {
    "id": "user123",
    "email": "student@university.edu",
    "name": "Jan Kowalski"
};

// Flow:
// 1. Security Agent logs query
// 2. Ticket Agent processes query
// 3. If ticket created:
//    - BOS Agent notified
//    - Analytics Agent logs statistics
// 4. User receives response
```

---

## 🔗 Quick Test Commands

### Test All Agents Quickly
```bash
# Agent 1
curl -X POST http://localhost:8001/run \
  -H "Content-Type: application/json" \
  -d '{"input": "Test query"}'

# Agent 2
curl -X POST http://localhost:8002/run \
  -H "Content-Type: application/json" \
  -d '{"input": "Test", "step": "initial"}'

# Agent 3
curl -X POST http://localhost:8003/run \
  -H "Content-Type: application/json" \
  -d '{"input": "Data: 1, 2, 3"}'

# Agent 4
curl -X POST http://localhost:8004/run \
  -H "Content-Type: application/json" \
  -d '{"input": "Test BOS"}'

# Agent 5
curl -X POST http://localhost:8005/run \
  -H "Content-Type: application/json" \
  -d '{"input": "Audit test"}'
```

### Health Checks
```bash
curl http://localhost:8001/health
curl http://localhost:8002/health
curl http://localhost:8003/health
curl http://localhost:8004/health
curl http://localhost:8005/health
```

---

## 📚 Related Files

- [Node-RED Flows](flows/)
- [Main README](../README.md)
- [Docker Commands](../COMMANDS.md)
