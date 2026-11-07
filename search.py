
from pathlib import Path

from qdrant_client import QdrantClient
from app.embedding import embeddings


# -----------------------------
# Configuration
# -----------------------------

BASE_DIR = Path(__file__).resolve().parent

DB_PATH = BASE_DIR / "data" / "qdrant_db"

COLLECTION_NAME = "neuromates_knowledge"

TOP_K = 10


# -----------------------------
# Semantic search
# -----------------------------

def search_knowledge(question):

    # 1. Convert question into an embedding
    query_vector = embeddings.embed_query(question)

    print("\nQuery:", question)
    print("Query vector dimensions:", len(query_vector))

    if len(query_vector) != 3072:
        raise ValueError(
            f"Unexpected query dimensions: {len(query_vector)}"
        )

    # 2. Connect to local Qdrant database
    client = QdrantClient(
        path=str(DB_PATH)
    )

    try:

        # 3. Search for similar vectors
        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=TOP_K,
            with_payload=True
        )

        # 4. Display results
        print("\nTop matching knowledge chunks")
        print("=" * 60)

        for rank, point in enumerate(results.points, start=1):

            payload = point.payload or {}

            print(f"\nRank: {rank}")
            print(f"Chunk ID: {payload.get('id', 'Unknown')}")
            print(f"Topic: {payload.get('topic', 'Unknown')}")
            print(f"Score: {point.score:.4f}")
            print(f"Content: {payload.get('content', 'No content')}")

            print("-" * 60)

        if not results.points:
            print("No matching knowledge found.")

    finally:
        client.close()


# -----------------------------
# Run search
# -----------------------------

if __name__ == "__main__":

    question = input(
        "Ask Neuro Mates something: "
    ).strip()

    if question:
        search_knowledge(question)
    else:
        print("Please enter a question.")
