from typing import Any, List, Dict

def build_input_parts(text: str, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert user text and images into chat input parts.
    """
    content = []
    if text and text.strip():
        content.append({"type": "input_text", "text": text.strip()})
    for img in images:
        content.append({"type": "input_image", "image_url": {"url": img["data_url"]}})
    return [{"type": "message", "role": "user", "content": content}] if content else []

def call_responses_api(
    system_prompt: str,
    client,
    model_name: str,
    embedding_model_name: str,
    parts: List[Dict[str, Any]],
    qdrant_client=None,
    qdrant_collection=None,
    top_k=5,
    previous_response_id=None
):
    """
    Generate AI response, optionally using Qdrant for context.
    """
    # Gather all user input text
    user_texts = [c["text"] for p in parts for c in p.get("content", []) if c.get("type") == "input_text"]
    user_prompt = "\n".join(user_texts)

    # Initialize context
    context_texts = []

    # If Qdrant is available, fetch similar context
    if qdrant_client and qdrant_collection:
        embedding = client.embeddings.create(model=embedding_model_name, input=user_prompt).data[0].embedding
        search_result = qdrant_client.query_points(
            collection_name=qdrant_collection,
            query=embedding,
            limit=top_k,
            with_payload=True
        ).points

        # Extract the text from Qdrant payloads
        context_texts = [point.payload.get("text", "") for point in search_result]

    # Join context into a single string
    context_str = '\n\n'.join([c for c in context_texts if c.strip()])

    # Prepare final prompt for the AI
    if context_str:
        final_input = f"{system_prompt}\n\nContext:\n{context_str}\n\nUser: {user_prompt}"
    else:
        final_input = f"{system_prompt}\n\nUser: {user_prompt}"

    # Call the AI
    return client.responses.create(
        model=model_name,
        input=final_input,
        previous_response_id=previous_response_id
    )

def get_text_output(response: Any) -> str:
    """
    Extract text output from API response.
    """
    return response.output_text

def embed_text(client, embedding_model_name, text: str) -> List[float]:
    """
    Generate vector embedding for text.
    """
    response = client.embeddings.create(model=embedding_model_name, input=text)
    return response.data[0].embedding

def embed_image(client, embedding_model_name, image_path) -> List[float]:
    """
    Embed image as pseudo-text using filename.
    """
    return embed_text(client, embedding_model_name, f"Image: {image_path.name}")
