
import json
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models


# --------------------------------
# Configuration
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "embedded_chunks.jsonl"
DB_PATH = BASE_DIR / "data" / "qdrant_db"

COLLECTION_NAME = "neuromates_knowledge"

VECTOR_DIMENSION = 3072
BATCH_SIZE = 20


# --------------------------------
# 1. Load embedded chunks
# --------------------------------

def load_embedded_chunks():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {INPUT_FILE}"
        )

    chunks = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        for line_number, line in enumerate(file, start=1):

            if not line.strip():
                continue

            record = json.loads(line)

            if not record.get("id"):
                raise ValueError(
                    f"Missing chunk ID at line {line_number}"
                )

            vector = record.get("embedding")

            if not isinstance(vector, list):
                raise ValueError(
                    f"Missing embedding for {record['id']}"
                )

            if len(vector) != VECTOR_DIMENSION:
                raise ValueError(
                    f"Incorrect dimensions for {record['id']}: "
                    f"{len(vector)}"
                )

            chunks.append(record)

    if not chunks:
        raise ValueError("No embedded chunks found!")

    ids = [chunk["id"] for chunk in chunks]

    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate chunk IDs found!")

    return chunks


# --------------------------------
# 2. Generate stable Qdrant IDs
# --------------------------------

def generate_point_id(chunk_id):

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"neuromates:{chunk_id}"
        )
    )


# --------------------------------
# 3. Create or validate collection
# --------------------------------

def create_collection(client):

    if client.collection_exists(COLLECTION_NAME):

        collection_info = client.get_collection(
            COLLECTION_NAME
        )

        existing_dimension = (
            collection_info.config.params.vectors.size
        )

        if existing_dimension != VECTOR_DIMENSION:
            raise ValueError(
                f"Collection dimension mismatch: "
                f"{existing_dimension}"
            )

        print("Existing Qdrant collection found.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=VECTOR_DIMENSION,
            distance=models.Distance.COSINE
        )
    )

    print("Qdrant collection created successfully!")


# --------------------------------
# 4. Upload vectors to Qdrant
# --------------------------------

def upload_vectors(client, chunks):

    for start in range(0, len(chunks), BATCH_SIZE):

        batch = chunks[start:start + BATCH_SIZE]

        points = []

        for chunk in batch:

            # Keep metadata, but don't duplicate
            # the embedding vector inside the payload.
            payload = {
                key: value
                for key, value in chunk.items()
                if key != "embedding"
            }

            point = models.PointStruct(
                id=generate_point_id(chunk["id"]),
                vector=chunk["embedding"],
                payload=payload
            )

            points.append(point)

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True
        )

        print(
            f"Uploaded {min(start + len(batch), len(chunks))}"
            f"/{len(chunks)} chunks"
        )


# --------------------------------
# 5. Verify stored records
# --------------------------------

def verify_collection(client, expected_count):

    result = client.count(
        collection_name=COLLECTION_NAME,
        exact=True
    )

    actual_count = result.count

    print("\nQdrant verification")
    print("-------------------")
    print("Expected records:", expected_count)
    print("Stored records:", actual_count)

    if actual_count != expected_count:
        raise ValueError(
            "Qdrant record count does not match!"
        )

    print("All records verified successfully!")


# --------------------------------
# Main execution
# --------------------------------

def main():

    chunks = load_embedded_chunks()

    print("Loaded chunks:", len(chunks))
    print("Vector dimensions:", VECTOR_DIMENSION)

    DB_PATH.mkdir(parents=True, exist_ok=True)

    client = QdrantClient(
        path=str(DB_PATH)
    )

    try:

        create_collection(client)

        upload_vectors(client, chunks)

        verify_collection(
            client,
            len(chunks)
        )

    finally:
        client.close()


if __name__ == "__main__":
    main()
