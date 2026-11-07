
import json
from pathlib import Path
import re

from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever


from app.embedding import embeddings


# --------------------------------
# Configuration
# --------------------------------

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "neuromates_rag_chunks.jsonl"
DB_PATH = BASE_DIR / "data" / "qdrant_db"

COLLECTION_NAME = "neuromates_knowledge"

TOP_K = 5
RRF_K = 60
CANDIDATE_K = 15
FINAL_TOP_K = 10

import re

def tokenize_text(text):
    text = text.lower()

    # Normalize apostrophes
    text = text.replace("’", "'").replace("‘", "'")

    # Extract words and numbers
    tokens = re.findall(r"[a-z0-9]+", text)

    # Remove common stopwords
    stopwords = {
        "the", "is", "a", "an", "and", "or",
        "of", "to", "in", "for", "how", "does",
        "do", "what", "which", "are", "was"
    }

    return [token for token in tokens if token not in stopwords]

MODULE_KEYWORDS = {
    "parkinson": "movement monitoring wearable ESP32 MPU6050 accelerometer gyroscope orientation unusual movement fall event alerts dorsal hand",
    "alzheimer": "wandering location safety GPS geofencing safe zones SIM800L alerts reminders panic button memory",
    "epilepsy": "EEG seizure detection BioAmp EXG headband electrode threshold emergency notification siren",
    "physiotherapy": "rehabilitation exercises muscle sensor EMG games motion recovery paralysis",
}

def expand_query(question):
    expanded_question = question.lower()
    for condition, keywords in MODULE_KEYWORDS.items():
        if condition in expanded_question:
            expanded_question += f" {keywords}"
    return expanded_question


# --------------------------------
# 1. Load documents for BM25
# --------------------------------


def load_documents():

    documents = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        for line in file:

            if not line.strip():
                continue

            chunk = json.loads(line)

            document = Document(
                page_content=chunk["content"],
                metadata={
                    key: value
                    for key, value in chunk.items()
                    if key != "embedding"
                }
            )

            documents.append(document)

    return documents



# --------------------------------
# 2. Build BM25 retriever
# --------------------------------

def create_bm25_retriever():

    documents = load_documents()

    retriever = BM25Retriever.from_documents(
    documents,
    k=CANDIDATE_K,
    preprocess_func=tokenize_text
)

    return retriever


# --------------------------------
# 3. Qdrant semantic retrieval
# --------------------------------

def dense_search(question):

    query_vector = embeddings.embed_query(question)

    if len(query_vector) != 3072:
        raise ValueError("Query embedding dimension mismatch!")

    client = QdrantClient(
        path=str(DB_PATH)
    )

    try:

        response = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=CANDIDATE_K,
            with_payload=True
        )

        results = []

        for point in response.points:

            payload = point.payload or {}

            results.append({
                "id": payload["id"],
                "content": payload.get("content", ""),
                "topic": payload.get("topic", ""),
                "score": point.score
            })

        cache_file = BASE_DIR / "data" / "dense_results.json"

        with open(cache_file, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "question": question,
                    "results": results
                },
                file,
                ensure_ascii=False,
                indent=2
            )

        print("Dense results saved!")

        return results

    finally:
        client.close()


# Cached in-memory retriever for performance (avoids reloading 45 docs on every query)
_CACHED_BM25_RETRIEVER = None

def get_bm25_retriever():
    global _CACHED_BM25_RETRIEVER
    if _CACHED_BM25_RETRIEVER is None:
        _CACHED_BM25_RETRIEVER = create_bm25_retriever()
    return _CACHED_BM25_RETRIEVER


# --------------------------------
# 4. Reciprocal Rank Fusion
# --------------------------------

def reciprocal_rank_fusion(dense_results, bm25_results, bm25_weight=2.0, dense_weight=1.0):

    fused_results = {}

    # Add dense search rankings
    for rank, result in enumerate(dense_results, start=1):

        chunk_id = result["id"]

        if chunk_id not in fused_results:
            fused_results[chunk_id] = {
                **result,
                "rrf_score": 0,
                "dense_rank": None,
                "bm25_rank": None
            }

        fused_results[chunk_id]["rrf_score"] += (
            dense_weight / (RRF_K + rank)
        )

        fused_results[chunk_id]["dense_rank"] = rank

    # Add BM25 rankings
    for rank, document in enumerate(bm25_results, start=1):

        chunk_id = document.metadata["id"]

        if chunk_id not in fused_results:

            fused_results[chunk_id] = {
                "id": chunk_id,
                "content": document.page_content,
                "topic": document.metadata.get("topic", ""),
                "rrf_score": 0,
                "dense_rank": None,
                "bm25_rank": None
            }

        fused_results[chunk_id]["rrf_score"] += (
            bm25_weight / (RRF_K + rank)
        )

        fused_results[chunk_id]["bm25_rank"] = rank

    # Sort by combined RRF score
    ranked_results = sorted(
        fused_results.values(),
        key=lambda item: item["rrf_score"],
        reverse=True
    )

    return ranked_results


# --------------------------------
# 5. Hybrid search
# --------------------------------

def hybrid_search(question, bm25_retriever=None, bm25_weight=2.0, dense_weight=1.0, top_k=FINAL_TOP_K):

    print("\nQuestion:", question)

    if bm25_retriever is None:
        bm25_retriever = get_bm25_retriever()

    # Stage 1: Dense retrieval (with resilient fallback if API/network is down)
    try:
        dense_results = dense_search(question)
    except Exception as error:
        print(f"\nDense search unavailable ({error}). Falling back to BM25 keyword retrieval.")
        dense_results = []

    # Stage 2: Keyword retrieval
    expanded_question = expand_query(question)

    bm25_results = bm25_retriever.invoke(expanded_question)

    print("\nDENSE SEARCH TOP RESULTS")

    for rank, result in enumerate(dense_results, start=1):
        print(rank, result["id"], result["topic"])

    print("\nBM25 SEARCH TOP RESULTS")

    for rank, document in enumerate(bm25_results, start=1):
        print(rank, document.metadata["id"], document.metadata.get("topic"))

    # Stage 3: Reciprocal Rank Fusion
    final_results = reciprocal_rank_fusion(
        dense_results,
        bm25_results,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight
    )

    # Stage 4: Select top candidates
    retrieved_chunks = final_results[:top_k]

    print("\nHYBRID RETRIEVAL RESULTS")
    print("=" * 65)

    for rank, result in enumerate(retrieved_chunks, start=1):

        print(f"\nRank: {rank}")
        print(f"Chunk ID: {result['id']}")
        print(f"Topic: {result['topic']}")
        print(f"RRF Score: {result['rrf_score']:.6f}")

        print("-" * 65)

    # Return chunks to the next RAG component.
    return retrieved_chunks



# --------------------------------
# Main
# --------------------------------


if __name__ == "__main__":

    bm25_retriever = create_bm25_retriever()

    print("BM25 index created!")
    print("Neuro Mates retrieval system ready!")

    question = input(
        "\nAsk Neuro Mates something: "
    ).strip()

    if question:

        retrieved_chunks = hybrid_search(
            question,
            bm25_retriever
        )

        print("\nRAG CONTEXT READY")
        print("Retrieved chunks:", len(retrieved_chunks))

    else:
        print("Please enter a question.")





