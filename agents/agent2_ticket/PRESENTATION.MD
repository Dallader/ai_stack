# Agent 2: Inteligentny Asystent Uczelniany z Systemem Zgłoszeń
## Prezentacja dla Klienta

---

## 🎯 Co to jest Agent 2?

**Agent 2** to inteligentny asystent wspierający studentów uczelni wyższej w codziennych sprawach administracyjnych. Działa jak pomocny pracownik biura obsługi studenta, który jest dostępny 24/7, rozumie pytania w naturalnym języku i potrafi automatycznie tworzyć zgłoszenia do odpowiednich działów.

### Dla kogo jest ten system?

- **Studenci** - mogą zadawać pytania o stypendia, praktyki, zaświadczenia, urlopy
- **Pracownicy BOS** - otrzymują uporządkowane, skategoryzowane zgłoszenia
- **Administracja uczelni** - może monitorować i analizować najczęstsze problemy studentów

---

## 💡 Główne Możliwości

### 1. **Inteligentna Rozmowa**
Agent rozumie pytania zadawane w naturalnym języku polskim:
- "Jak mogę uzyskać stypendium socjalne?"
- "Gdzie składam podanie o urlop dziekański?"
- "Potrzebuję zaświadczenia o studiowaniu"

**Jak to działa?**
System wykorzystuje sztuczną inteligencję GPT-5, która została przeszkolona na dokumentach uczelni. Nie wymyśla odpowiedzi - zawsze opiera się na oficjalnych procedurach i regulaminach.

### 2. **Wyszukiwanie w Bazie Wiedzy**
System posiada dostęp do szerokiej bazy dokumentów uczelnianych:
- Procedury dotyczące stypendiów
- Harmonogramy obron
- Wzory podań
- Informacje o praktykach (dla każdego wydziału osobno)
- Procedury dotyczące zaświadczeń
- Udogodnienia dla osób z niepełnosprawnościami

**Technologia RAG (Retrieval-Augmented Generation)**
Zamiast odpowiadać na podstawie ogólnej wiedzy, system najpierw przeszukuje swoją bazę dokumentów, znajduje najważniejsze fragmenty i dopiero na ich podstawie formułuje odpowiedź. To gwarantuje aktualność i zgodność z wewnętrznymi procedurami uczelni.

### 3. **Automatyczne Tworzenie Zgłoszeń**
Gdy student potrzebuje konkretnej pomocy, system może automatycznie utworzyć zgłoszenie do BOS:

**Proces tworzenia zgłoszenia:**
1. System rozpoznaje, że student chce złożyć wniosek
2. Zbiera niezbędne dane (imię, nazwisko, email, nr indeksu)
3. Automatycznie kategoryzuje zgłoszenie (np. "Dokumenty finansowe", "Prośby o stypendia")
4. Przypisuje priorytet (Pilne, Ważne, Informacyjne)
5. Kieruje do właściwego działu
6. Generuje unikalny numer zgłoszenia

**Przykładowe kategorie zgłoszeń:**
- Wnioski o przyjęcie na studia
- Zmiana danych osobowych
- Prośby o urlop lub zawieszenie
- Prośby o stypendia i świadczenia
- Dokumenty dotyczące zaliczeń i egzaminów
- Reklamacje i skargi
- Dokumenty finansowe
- Zaświadczenia i potwierdzenia
- Prośby o wsparcie administracyjne lub techniczne

### 4. **Obsługa Dokumentów i Zdjęć**
Student może załączyć do rozmowy:
- Zdjęcia dokumentów (JPG, PNG, WebP)
- Pliki PDF
- Dokumenty Word i Excel
- Pliki tekstowe

System automatycznie:
- Analizuje zawartość dokumentów
- Ekstrahuje tekst (nawet ze zeskanowanych dokumentów dzięki OCR)
- Dodaje dokumenty do bazy wiedzy
- Wykorzystuje je w kontekście dalszej rozmowy

### 5. **Pamięć Konwersacji**
System pamięta całą historię rozmowy, dzięki czemu:
- Nie trzeba powtarzać informacji
- Odpowiedzi są kontekstowe
- Można zadawać pytania uzupełniające
- Student może łatwo wrócić do wcześniejszych tematów

---

## 🎨 Interfejs Użytkownika

### Prosty i Intuicyjny Chat
- **Okno rozmowy** - przejrzysta historia wszystkich wiadomości
- **Pole tekstowe** - miejsce na wpisanie pytania
- **Przycisk załączników** - możliwość dodawania plików
- **Przycisk czyszczenia** - reset rozmowy i rozpoczęcie od nowa

### Panel Informacyjny (opcjonalny dla administratorów)
- Status bazy wiedzy
- Liczba dokumentów do zaimportowania
- Statystyki poszczególnych kolekcji
- Przycisk importu nowych dokumentów

---

## 🔒 Bezpieczeństwo i Prywatność

- **Dane osobowe** - wszystkie dane studentów są przetwarzane zgodnie z RODO
- **Bezpieczne połączenia** - komunikacja szyfrowana
- **Kontrola dostępu** - możliwość integracji z systemem uwierzytelniania uczelni
- **Logi działań** - pełna historia operacji dla audytu

---

## 📊 Korzyści dla Uczelni

### Dla Studentów:
✅ Dostęp do informacji 24/7  
✅ Natychmiastowe odpowiedzi na pytania  
✅ Łatwe składanie wniosków  
✅ Brak kolejek i czasu oczekiwania  
✅ Dostęp z dowolnego urządzenia (komputer, telefon, tablet)

### Dla Pracowników BOS:
✅ Mniej powtarzalnych pytań  
✅ Uporządkowane zgłoszenia z pełnym kontekstem  
✅ Automatyczna kategoryzacja i priorytetyzacja  
✅ Więcej czasu na sprawy wymagające ludzkiej interwencji  
✅ Pełna historia rozmowy ze studentem

### Dla Uczelni:
✅ Lepsza jakość obsługi studenta  
✅ Obniżenie kosztów operacyjnych  
✅ Dostęp do analityki (najczęstsze pytania, problemy)  
✅ Nowoczesny wizerunek uczelni  
✅ Skalowalność - system obsługuje wielu użytkowników jednocześnie

---

## 🚀 Możliwości Rozbudowy

System został zaprojektowany modularnie, co oznacza możliwość łatwego dodawania nowych funkcji:

### Planowane Rozszerzenia:
- **Integracja z systemem USOS** - automatyczne pobieranie danych studenta
- **Powiadomienia email/SMS** - status zgłoszenia
- **Panel administratora** - zarządzanie zgłoszeniami
- **Wielojęzyczność** - obsługa studentów zagranicznych
- **Analityka zaawansowana** - raporty i dashboardy
- **Asystent głosowy** - rozmowa zamiast pisania

---

## 📈 Przykładowe Scenariusze Użycia

### Scenariusz 1: Student pyta o stypendium
**Student:** "Czy mogę dostać stypendium?"  
**Agent:** Przedstawia przegląd dostępnych stypendiów (socjalne, rektora, dla sportowców, dla osób z niepełnosprawnościami, Erasmus), wyjaśnia warunki i procedury składania wniosków.

### Scenariusz 2: Zmiana danych osobowych
**Student:** "Chcę zmienić adres email na nowy@example.com"  
**Agent:** 
1. Prosi o potwierdzenie danych osobowych (imię, nazwisko, nr indeksu)
2. Tworzy zgłoszenie kategorii "Zmiana danych osobowych"
3. Przypisuje priorytet "Ważne"
4. Generuje numer zgłoszenia
5. Informuje studenta o dalszych krokach

### Scenariusz 3: Pytanie o praktyki
**Student (Wydział Informatyki):** "Jak załatwić praktyki?"  
**Agent:** Wyszukuje informacje specyficzne dla Wydziału Informatyki i Nowych Technologii, przedstawia procedury, terminy i wymagane dokumenty.

### Scenariusz 4: Załączenie dokumentu
**Student:** Załącza skan podania i pyta "Czy to podanie jest wypełnione prawidłowo?"  
**Agent:** Analizuje dokument, sprawdza completeness i sugeruje ewentualne poprawki lub potwierdza poprawność.

---

## 💻 Szczegóły Techniczne (dla programistów)

### Architektura Systemu

```
┌────────────────────────────────────────────┐
│           Interfejs Użytkownika            │
│              (Streamlit)                   │
└──────────────────┬─────────────────────────┘
                   │
┌──────────────────▼─────────────────────────┐
│           Warstwa Aplikacyjna              │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │  Moduł Konwersacji (app.py)          │  │
│  │  - Zarządzanie sesją                 │  │
│  │  - Historia rozmów                   │  │
│  │  - Obsługa uploads                   │  │
│  └──────────────────────────────────────┘  │
│                                            |
│  ┌──────────────────────────────────────┐  │
│  │  Moduł Ticketów (ticket.py)          │  │
│  │  - Tworzenie zgłoszeń                │  │
│  │  - Kategoryzacja                     │  │
│  │  - Priorytetyzacja                   │  │
│  └──────────────────────────────────────┘  │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │  Moduł RAG (api_utils.py)            │  │
│  │  - Embeddingi                        │  │
│  │  - Wyszukiwanie semantyczne          │  │
│  │  - Kontekst z dokumentów             │  │
│  └──────────────────────────────────────┘  │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │  Moduł Qdrant (qdrant_utils.py)      │  │
│  │  - Zarządzanie kolekcjami            │  │
│  │  - Import dokumentów                 │  │
│  │  - Chunking i deduplikacja           │  │
│  └──────────────────────────────────────┘  │
└──────────────────┬─────────────────────────┘
                   │
┌──────────────────▼─────────────────────────┐
│          Warstwa Infrastruktury            │
│                                            │
│  ┌──────────────┐      ┌────────────────┐  │
│  │   OpenAI     │      │    Qdrant      │  │
│  │   GPT-5      │      │  Vector DB     │  │
│  │  Responses   │      │  Port: 6333    │  │
│  │     API      │      │                │  │
│  └──────────────┘      └────────────────┘  │
└────────────────────────────────────────────┘
```

### Stack Technologiczny

#### Frontend & UI
- **Streamlit** - framework do tworzenia interaktywnych aplikacji webowych w Pythonie
- **Custom CSS** - personalizacja wyglądu interfejsu

#### Backend & Logika
- **Python 3.10+** - język programowania
- **OpenAI Responses API** - konwersacyjny interfejs z GPT-5
- **Dataclasses** - struktury danych dla ticketów

#### Baza Wiedzy & Wyszukiwanie
- **Qdrant** - wektorowa baza danych do wyszukiwania semantycznego
- **text-embedding-3-large** - model embeddingów OpenAI
- **RAG (Retrieval-Augmented Generation)** - technika łącząca wyszukiwanie z generowaniem odpowiedzi

#### Przetwarzanie Dokumentów
- **pdf2image** + **Poppler** - konwersja PDF na obrazy
- **pytesseract** + **Tesseract OCR** - ekstrakcja tekstu ze zeskanowanych dokumentów
- **python-docx** - obsługa plików Word
- **python-dotenv** - zarządzanie zmiennymi środowiskowymi

#### Deployment
- **Docker** - konteneryzacja aplikacji
- **docker-compose** - orkiestracja kontenerów
- **Port 8002** - domyślny port usługi

### Kolekcje w Qdrant

System wykorzystuje dwie główne kolekcje wektorowe:

1. **Documents** - dokumenty z bazy wiedzy
   - Zaświadczenia
   - Stypendia
   - Praktyki
   - Podania
   - Obrony
   - Inne dokumenty akademickie

2. **Tickets** - historia zgłoszeń
   - Przechowuje utworzone tickety
   - Umożliwia analizę historyczną
   - Kontekst dla przyszłych rozmów

### Proces RAG (Retrieval-Augmented Generation)

```python
# Uproszczony flow:
1. Użytkownik zadaje pytanie: "Jak uzyskać stypendium?"

2. System tworzy embedding pytania:
   embedding = embed_text(query, model="text-embedding-3-large")

3. Wyszukiwanie w Qdrant (top-k=5):
   results = qdrant_client.search(
       collection="Documents",
       query_vector=embedding,
       limit=5
   )

4. Budowa kontekstu z najlepszych wyników:
   context = "\n".join([r.payload["text"] for r in results])

5. Wywołanie GPT-5 z kontekstem:
   response = client.responses.create(
       model="gpt-5.1",
       instructions=system_prompt + context,
       input=user_message
   )

6. Zwrócenie odpowiedzi do użytkownika
```

### Automatyczna Kategoryzacja Ticketów

System używa LLM do inteligentnego przypisywania kategorii:

```python
def assign_category_priority_department(
    client: OpenAI, 
    model_name: str,
    conversation_context: str
) -> Dict[str, str]:
    """
    LLM analizuje kontekst rozmowy i wybiera:
    - Kategorie z predefiniowanej listy
    - Priorytet (Pilne/Ważne/Informacyjne)
    - Departament (Wydział/Dział)
    
    Zwraca JSON:
    {
        "category": "Prośby o stypendia i świadczenia",
        "priority": "Ważne",
        "department": "Biuro Finansowe"
    }
    """
```

### Import i Indeksowanie Dokumentów

System automatycznie przetwarza dokumenty:

1. **Haszowanie (SHA256)** - deduplikacja, pomijanie już przetworzonych plików
2. **Ekstrakcja tekstu**:
   - Dokumenty tekstowe → bezpośrednie czytanie
   - PDF → pdfplumber (text layer) lub OCR (scanned)
   - Word/Excel → biblioteki python-docx/openpyxl
   - Obrazy → Tesseract OCR
3. **Chunking** - podział na fragmenty (800 znaków, 200 overlap)
4. **Embedding** - konwersja na wektory
5. **Upsert do Qdrant** - zapis z metadanymi (źródło, kategoria, timestamp)

### Konfiguracja (settings/*.json)

#### llm.json
Definiuje personalność i zachowanie asystenta:
- Persona (rola systemowa)
- Cele (jak odpowiadać)
- Ograniczenia (czego unikać)
- Zasady formatowania (Markdown, bold, listy, tabele)

#### categories.json
Lista dostępnych kategorii zgłoszeń BOS.

#### priorities.json
Poziomy priorytetów dla ticketów.

#### departments.json
Działy/wydziały, do których mogą być kierowane zgłoszenia.

#### models.json
Referencyjne modele AI dostępne w systemie.

#### qdrant.json
Konfiguracja kolekcji wektorowych (nazwy, wymiary).

### Zmienne Środowiskowe

```bash
# OpenAI
OPENAI_API_KEY=sk-xxxxx                   # Klucz API OpenAI
MODEL_NAME=gpt-5.1                        # Model dla rozmów
CLASSIFYING_MODEL_NAME=gpt-5.1            # Model dla kategoryzacji
EMBEDING_MODEL_NAME=text-embedding-3-large # Model embeddingów

# Qdrant
QDRANT_URL=http://qdrant:6333             # URL do Qdrant

# UI
SIDEBAR=true                               # Pokazuj panel administratora
DEBUG=false                                # Tryb debug (ścieżki lokalne vs kontener)
```

### Instalacja Zależności (Requirements)

```txt
streamlit                    # Framework UI
openai                       # OpenAI API client
qdrant-client               # Qdrant vector DB client
python-dotenv               # Zarządzanie .env
pdf2image                   # Konwersja PDF → obrazy
pytesseract                 # OCR dla Tesseract
python-docx                 # Obsługa dokumentów Word
Pillow                      # Przetwarzanie obrazów
pdfplumber                  # Ekstrakcja tekstu z PDF
```

### Dockerfile & Deployment

```dockerfile
FROM python:3.10-slim

# Install system dependencies dla PDF/OCR
RUN apt-get update && apt-get install -y \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-pol \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . /app
WORKDIR /app

# Expose port
EXPOSE 8002

# Run Streamlit
CMD ["streamlit", "run", "app.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8002"]
```

### Integracja z Docker Stack

Agent 2 jest częścią większego systemu multi-agentowego:
- **Uruchamiany przez docker-compose** z głównego katalogu
- **Komunikuje się z Qdrant** (port 6333)
- **Może współpracować z innymi agentami** przez Node-RED
- **Persystentne dane** w wolumenach Docker

### API Endpoints (wewnętrzne)

Chociaż główny interfejs to Streamlit, można rozbudować o REST API do integracji:

```python
# Potencjalne API (przykład)
POST /api/chat
  Body: { "message": "Pytanie studenta", "session_id": "xxx" }
  Response: { "response": "Odpowiedź asystenta", "response_id": "yyy" }

POST /api/ticket
  Body: { "description": "...", "student_data": {...} }
  Response: { "ticket_id": "zzz", "category": "...", "priority": "..." }

GET /api/documents/stats
  Response: { "total": 1234, "by_category": {...} }
```

### Rozszerzalność

Kod jest modularny i łatwy do rozbudowy:

#### Dodawanie nowych kategorii
Edycja `settings/categories.json`

#### Dodawanie nowych typów dokumentów
Rozszerzenie funkcji `extract_text_from_pdf()` w `file_utils.py`

#### Nowe języki
Zmiana prompt w `llm.json`, dodanie language pack do Tesseract

#### Integracja z zewnętrznymi systemami
Dodanie klientów API w nowych modułach (np. `usos_utils.py`)

### Monitoring i Logi

```python
# Obecne logowanie
documents/logs/              # Katalog na logi (do implementacji)

# Możliwe rozszerzenia:
- Structured logging (JSON)
- ELK Stack integration
- Prometheus metrics
- Grafana dashboards
```

### Testy i Jakość

#### Obecne podejście:
- Manualne testy funkcjonalne
- Weryfikacja odpowiedzi na przykładowych pytaniach

#### Potencjalne rozszerzenia:
```python
# Unit tests
pytest tests/test_ticket.py
pytest tests/test_qdrant_utils.py

# Integration tests
pytest tests/integration/test_chat_flow.py

# LLM evaluation
- RAGAS metrics (faithfulness, relevancy)
- Human feedback loop
```

---

## 📋 Wymagania Systemowe

### Minimalne:
- **RAM**: 2 GB (sam agent)
- **CPU**: 2 rdzenie
- **Dysk**: 5 GB
- **Docker**: 20.10+

### Zalecane (pełny stack z Qdrant):
- **RAM**: 8 GB
- **CPU**: 4 rdzenie
- **Dysk**: 20 GB
- **Docker Desktop**: Aktualna wersja

---

## 🎓 Podsumowanie

**Agent 2** to kompleksowe rozwiązanie wspierające obsługę studenta na uczelni wyższej. Łączy w sobie:
- 🤖 Najnowszą technologię AI (GPT-5)
- 📚 Wyszukiwanie semantyczne w bazie wiedzy
- 🎫 Automatyzację procesów ticketowych
- 💬 Intuicyjny interfejs konwersacyjny
- 🔒 Bezpieczeństwo i zgodność z RODO

System jest **gotowy do wdrożenia**, **łatwy w konfiguracji** i **otwarty na rozszerzenia** zgodnie z potrzebami konkretnej uczelni.

---

## 📞 Kontakt i Dalsze Kroki

Aby wdrożyć system na swojej uczelni lub uzyskać więcej informacji:
1. **Demo na żywo** - zapraszamy do przetestowania systemu
2. **Konsultacja** - omówienie specyficznych potrzeb uczelni
3. **Dostosowanie** - personalizacja kategorii, działów, procedur
4. **Wdrożenie** - instalacja i szkolenie pracowników
5. **Wsparcie** - bieżące utrzymanie i rozwój systemu

---

*Dokument wygenerowany: Luty 2026*  
*Wersja Agenta: 2.0*  
*Status: Produkcyjny*
