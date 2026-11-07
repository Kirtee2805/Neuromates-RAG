
import json
import time
from pathlib import Path

from app.embedding import embeddings

# -----------------------------
# Project configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "neuromates_rag_chunks.jsonl"
OUTPUT_FILE = BASE_DIR / "data" / "embedded_chunks.jsonl"

EXPECTED_DIMENSION = 3072
PAUSE_SECONDS = 0.5

EMBEDDING_MODEL = getattr(
    embeddings,
    "model",
    "configured-gemini-embedding-model"
)


# -----------------------------
# 1. Load and validate chunks
# -----------------------------

def load_knowledge_chunks():
    """Load and validate knowledge chunks from JSONL."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge file not found: {INPUT_FILE}"
        )

    chunks = []
    seen_ids = set()

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                chunk = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at line {line_number}: {error}"
                )

            if not isinstance(chunk, dict):
                raise ValueError(
                    f"Expected JSON object at line {line_number}"
                )

            if not chunk.get("id") or not chunk.get("content"):
                raise ValueError(
                    f"Missing id/content at line {line_number}"
                )

            chunk_id = chunk["id"]

            if chunk_id in seen_ids:
                raise ValueError(
                    f"Duplicate chunk ID found: {chunk_id}"
                )

            seen_ids.add(chunk_id)
            chunks.append(chunk)

    return chunks


# -----------------------------
# 2. Check completed embeddings
# -----------------------------

def get_completed_ids():
    """Load valid embeddings already saved."""

    completed_ids = set()

    if not OUTPUT_FILE.exists():
        return completed_ids

    with open(OUTPUT_FILE, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(
                    f"Invalid JSON in output at line {line_number}. "
                    "Inspect embedded_chunks.jsonl before continuing."
                )

            chunk_id = record.get("id")
            vector = record.get("embedding")

            if not chunk_id or not isinstance(vector, list):
                raise ValueError(
                    f"Invalid embedding record at line {line_number}"
                )

            if len(vector) != EXPECTED_DIMENSION:
                raise ValueError(
                    f"Incorrect dimensions for {chunk_id}: "
                    f"{len(vector)}"
                )

            completed_ids.add(chunk_id)

    return completed_ids


# -----------------------------
# 3. Generate embeddings
# -----------------------------

def generate_embeddings():

    chunks = load_knowledge_chunks()

    print("=" * 50)
    print("NEURO MATES - EMBEDDING INGESTION")
    print("=" * 50)

    print("Total knowledge chunks:", len(chunks))
    print("Embedding model:", EMBEDDING_MODEL)
    print("Expected dimensions:", EXPECTED_DIMENSION)

    completed_ids = get_completed_ids()

    print("Already embedded:", len(completed_ids))

    pending_chunks = [
        chunk for chunk in chunks
        if chunk["id"] not in completed_ids
    ]

    print("Remaining chunks:", len(pending_chunks))

    if not pending_chunks:
        print("\nAll chunks are already embedded!")
        return

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(OUTPUT_FILE, "a", encoding="utf-8") as output:

        for index, chunk in enumerate(pending_chunks, start=1):

            document_text = (
                f"title: {chunk.get('topic', 'none')} | "
                f"text: {chunk['content']}"
            )

            print(
                f"\n[{index}/{len(pending_chunks)}] "
                f"Embedding {chunk['id']}"
            )

            vector = embeddings.embed_documents(
                [document_text]
            )[0]

            if len(vector) != EXPECTED_DIMENSION:
                raise ValueError(
                    f"Unexpected dimension for {chunk['id']}: "
                    f"{len(vector)}"
                )

            record = {
                **chunk,
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimension": len(vector),
                "embedding": vector
            }

            output.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )

            output.flush()

            print(
                f"Saved successfully: {chunk['id']} "
                f"({len(vector)} dimensions)"
            )

            time.sleep(PAUSE_SECONDS)

    print("\n" + "=" * 50)
    print("EMBEDDING GENERATION COMPLETED")
    print("=" * 50)

    print("Total source chunks:", len(chunks))
    print("Total embedded chunks:", len(get_completed_ids()))
    print("Output file:", OUTPUT_FILE)


if __name__ == "__main__":
    generate_embeddings()
