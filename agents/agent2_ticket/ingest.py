"""CLI ingestion script to index agent2_ticket documents into Qdrant.

Usage:
    python ingest.py

Relies on the same logic as the FastAPI app: reads knowledge/, incoming/, documents/ (legacy),
and processed_documents/, extracts text, chunks, embeds with the configured model, and upserts to Qdrant.
"""

from app import ensure_collection_exists, load_and_index_documents


def main() -> None:
    ensure_collection_exists()
    chunks = load_and_index_documents()
    if chunks:
        print(f"Indexed {chunks} chunks into collection")
    else:
        print("No new documents to index (all existing or no files found)")


if __name__ == "__main__":
    main()