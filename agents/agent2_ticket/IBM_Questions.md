# Analiza Agenta 2 - Odpowiedzi na Pytania IBM

---

## 1. **Rozgraniczenie System Prompt vs Baza Wiedzy**

Podejście jest **jasno rozdzielone**:

**System Prompt** [`settings/llm.json`](agents/agent2_ticket/settings/llm.json):
- Zawiera **instrukcje behawioralne**: persona ("helpful university assistant"), ograniczenia ("Do not make up any information"), zasady formatowania Markdown
- Krótki i uniwersalny — unika duplikacji wiedzy domenowej
- Wspólny dla wszystkich rozmów

**Baza Wiedzy** → Qdrant collection `Documents`:
- Przechowuje faktyczne dokumenty uczelni (procedury, regulaminy, stypendia itp.)
- Indexowane wektorowo za pomocą embeddings
- Dynamicznie dodawane/aktualizowane poprzez import z `documents/knowledge/`
- Potrójnie deduplikowana: SHA256 hash + semantic search + categorization

**Integracja w RAG flow** [`tools/api_utils.py`](agents/agent2_ticket/tools/api_utils.py):

```python
# 1. Pytanie użytkownika → embedding
embedding = client.embeddings.create(model=embedding_model_name, input=user_prompt)

# 2. Wyszukanie top-5 dokumentów z Qdrant
context_texts = qdrant_client.query_points(
    collection_name="Documents",
    query=embedding,
    limit=5  # ← optymalizacja!
).points

# 3. Dołączenie kontekstu do system prompta
final_input = f"{system_prompt}\n\nContext:\n{context_str}\n\nUser: {user_prompt}"
```

**Korzyść**: System prompt pozostaje czysty i uniwersalny, a wiedza jest wersjonowana/auditowana w bazie.

---

## 2. **Podejście do Testów i Automatyzacji**

**Status quo** — brak formalnych testów automatycznych (żaden `test_*.py` w repozytorium).

**Aktualne praktyki obserwacyjne**:

- **Logi strukturyzowane w JSON** [`tools/logs_utils.py`](agents/agent2_ticket/tools/logs_utils.py):
  ```python
  # documents/logs/chat_log_YYYY-MM-DD.json
  {
    "timestamp": "2026-02-16T10:30:00",
    "event_type": "user_message|assistant_message|ticket_created|error",
    "data": {...}
  }
  ```

- **Obserwacja na żywo** — Streamlit sidebar pokazuje:
  - Ilość plików do importu
  - Statystyki kolekcji Qdrant
  - Liczba kategorii w bazie

**Brakuje automatyzacji**:
- ❌ Unit tests (pytest dla modułów)
- ❌ Testy regresji (RAGAS metrics dla RAG faithful)
- ❌ CI/CD pipeline
- ❌ Benchmark dla quality odpowiedzi

**Propozycja rozbudowy** (w dokumentacji):
```python
# Potencjalne
- RAGAS metrics (faithfulness, relevancy)
- Human feedback loop
- LLM evaluation prompts
```

---

## 3. **Optymalizacja Użycia Tokenów**

Strategie implementowane:

| Strategia | Implementacja |
|-----------|---------------|
| **Chunking** | 1000 znaków ze wspólnym fragmentem (overlap) 200 w [`qdrant_utils.py`](agents/agent2_ticket/tools/qdrant_utils.py) |
| **Top-K Limiting** | `top_k=5` zamiast wyszukiwania całej bazy w [`api_utils.py`](agents/agent2_ticket/tools/api_utils.py) |
| **Odpowiedzi krótkie** | System prompt mówi: "5-6 sentences" → ogranicza rozwlekłe odpowiedzi |
| **Deduplikacja** | SHA256 hashing unika importowania duplikatów |
| **Konwersacja zwista** | `previous_response_id` w Responses API — przechowuję kontekst tura po turze zamiast wysyłać całą historię |
| **Selective Context** | Tylko pytanie użytkownika jest embeddowane i szukane w bazie — obrazy kodowane są jako pseudo-embeddingi |

**Wynik**: Dla typowego pytania studenta:
- System prompt: ~200 tokenów
- RAG context (5 chunks × 200 tokens avg): ~1000 tokenów
- Pytanie: ~50 tokenów
- **Total input ≈ 1300 tokenów** (zamiast potencjalnie 5000+)

---

## 4. **Balans Halucynacja vs Alienacja (Naturalna Komunikacja)**

**Zapobieganie halucynacji**:

1. **RAG grounding** — odpowiadają TYLKO na bazie dokumentów w Qdrant
2. **Explicit constraint**: `"Do not make up any information"` w system prompt
3. **Fallback graceful**: Jeśli nie znają odpowiedzi → polecają ticket:
   ```python
   # Z llm.json
   "If you do not know the answer, you politely ask for more details; 
    if still uncertain, suggest creating a ticket in the BOS."
   ```

**Balans naturalności vs alienacji**:

- **Persona**: `"You are a helpful university assistant"` — nie `"You are a chatbot"`
- **Formatowanie Markdown**: System prompt wymaga struktury:
  - Hierarchiczne nagłówki (##, ###)
  - Punkty listy dla przejrzystości
  - Tabele dla porównań
  - Cytowania dla definicji i wskazówek
- **Greetings only once**: `"Greet the user only once at beginning, thereafter respond naturally"`
- **Conversational continuity**: `previous_response_id` w Responses API utrzymuje spójność tonu między kolejnymi turami rozmowy
- **Contextual awareness**: Znają historię rozmowy z `st.session_state.messages`

**Wynik**: Agent brzmi jak rzeczywisty pracownik BOS z dostępem do instrukcji, a nie jak maszyna.

---

## 5. **Dialog Pogłębiający Wiedzę (Nie Jedno-Promptowe)**

**Multi-turn conversation architecture**:

1. **Full conversation memory** [`tools/llm_utils.py`](agents/agent2_ticket/tools/llm_utils.py):
   ```python
   def get_full_conversation(messages) -> str:
       """Zwraca pełną rozmowę użytkownika jako jeden string"""
       # Iteruje przez st.session_state.messages
   ```

2. **Context-aware responses**:
   - Wiele rzeczy w ten sam turze (obrazy + tekst + poprzednia historia)
   - Responses API zapisuje `previous_response_id` — agent wie o poprzednich swoich odpowiedziach

3. **Smart ticket creation**:
   ```python
   # Kiedy student chce zgłoszenie, system:
   1. Zbiera dane (imię, email, nr indeksu)
   2. Podsumowuje całą rozmowę: summarize_conversation()
   3. Tworzy szczegółowy opis z pełnym kontekstem
   ```

4. **Follow-up intelligence**:
   - Pytanie → RAG context
   - Odpowiedź
   - Follow-up pytanie student → nowy RAG query (nie powtarzanie poprzedniego kontekstu)
   - Agent rozumie progresję rozmowy z `previous_response_id`

5. **Category & Priority extraction**:
   ```python
   # Inteligentne przypisanie, a nie zwykły matching — pełna analiza LLM:
   def assign_category_priority_department(
       client, model_name, conversation_context
   ) -> Dict:
       # "na podstawie CAŁEJ rozmowy wybierz kategorię, priorytet i dział"
   ```

**Wynik**: Student może prowadzić pogłębiającą rozmowę, gdzie agent rozumie:
- Kontekst poprzednich odpowiedzi
- Ewolucję pytania
- Potrzebę dodatkowych szczegółów
- Odpowiednią eskalację do ticketu

---

## **TL;DR — Filozofia Agent 2**

| Aspekt | Podejście |
|--------|-----------|
| **Wiedza** | Oddzielona: prompt = instrukcje, Qdrant = fakty |
| **Halucynacja** | RAG + explicit constraints + graceful fallback |
| **Tokeny** | Chunking, top-k, krótkie odpowiedzi, context reuse |
| **Naturalność** | Persona, formatowanie, jedno przywitanie, spójność w rozmowie |
| **Inteligencja** | Pamięć wieloturowa, pełny kontekst rozmowy, inteligentna kategoryzacja przez LLM |
| **Testy** | Brak formalnych testów (potencjał do rozbudowy!) — logi JSON dla audytu |

System jest **production-ready**, ale mógłby skorzystać z formalnego test suite'u (RAGAS, pytest) i CI/CD pipeline.
