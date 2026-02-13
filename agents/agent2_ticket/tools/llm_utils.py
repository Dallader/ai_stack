import json 

def determine_category_llm(client, text, categories, classifying_model_name):
    """Determine document category using LLM."""
    try:
        categories_str = "\n".join(f"- {c}" for c in categories)
        prompt = (
            "You are a document classifier. Assign text to exactly one category. Respond with only the category name.\n"
            f"Categories:\n{categories_str}\n\nText:\n{text[:2000]}...\nCategory:"
        )
        response = client.responses.create(model=classifying_model_name, input=prompt, temperature=0)
        category = response.output_text.strip()
        if category not in categories:
            return "Pozostałe dokumenty"
        return category
    except Exception: 
        return "Pozostałe dokumenty"


def build_system_prompt(prompt_dict):
    """
    Flatten system prompt dict into formatted string.
    """
    parts = []
    for key, value in prompt_dict.items():
        if isinstance(value, dict):
            parts.append(f"{key.capitalize()}:")
            for subkey, subvalue in value.items():
                parts.append(f"- {subkey}: {subvalue}")
        else:
            parts.append(f"{key.capitalize()}: {value}")
    return "\n".join(parts)

def should_create_ticket(user_prompt: str, assistant_response: str) -> bool:
    """
    Decide whether to create a ticket.
    - True if user asks in any natural way to create a ticket.
    - True if assistant indicates it cannot answer confidently.
    """

    ticket_keywords = [
        "ticket",
        "zgłoszenie",
        "utwórz ticket",
        "napisz ticket",
        "stwórz zgłoszenie",
        "potrzebuję pomocy",
        "złożyć ticket",
        "wyślij zgłoszenie",
        "utworzyć zgłoszenie"
    ]
    
    user_prompt_lower = user_prompt.lower()
    user_intent = any(keyword in user_prompt_lower for keyword in ticket_keywords)
    
    system_keywords = ["cannot answer", "nie mogę odpowiedzieć", "brak informacji", "nie znalazłem odpowiedzi", "suggest creating a ticket"]
    system_suggestion = any(keyword in assistant_response.lower() for keyword in system_keywords)

    return user_intent or system_suggestion

def extract_change_request(client, model_name, prompt: str) -> dict:
    """
    Extracts change request details from user prompt.
    """
    system_prompt = """
    You extract structured data from a student's request to BOS.
    If information is missing, return null for that field.

    Respond ONLY in JSON:
    {
      "field": "<what is being changed>",
      "old_value": "<current value or null>",
      "new_value": "<new value or null>"
    }
    """

    response = client.responses.create(
        model=model_name,
        input=[
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}]
            }
        ],
        instructions=system_prompt,
        temperature=0
    )

    try:
        return json.loads(response.output_text)
    except Exception:
        return {"field": None, "old_value": None, "new_value": None}

def get_full_conversation(messages) -> str:
    """
    Zwraca pełną rozmowę użytkownika jako jeden string,
    z listy wiadomości Streamlit session_state['messages'].
    """
    parts = []

    for msg in messages:
        if msg.get("role") == "user":
            content = msg.get("content")
            if isinstance(content, list):
                for part in content:
                    for content_item in part.get("content", []):
                        if content_item.get("type") == "input_text":
                            parts.append(content_item["text"])
            elif isinstance(content, str):
                parts.append(content)

    full_conversation = "\n".join(parts).strip()
    return full_conversation if full_conversation else "Brak treści rozmowy"


def summarize_conversation(client, model_name, messages) -> str:
    """
    Podsumowuje całą rozmowę użytkownika w celu użycia w ticket.description
    """
    full_text = get_full_conversation(messages)
    if not full_text.strip():
        return "Brak treści rozmowy"

    prompt = f"""
    Podsumuj poniższą rozmowę użytkownika w formie krótkiego opisu zgłoszenia do BOS.
    Zachowaj wszystkie istotne informacje, nie powtarzaj dokładnie wszystkich wiadomości.

    {full_text}
    """

    response = client.responses.create(
        model=model_name,
        input=[{"type":"message","role":"user","content":[{"type":"input_text","text":prompt}]}]
    )

    try:
        return response.output_text.strip()
    except Exception:
        return full_text[:500]  # fallback