import json
from pathlib import Path
from hybrid_search import create_bm25_retriever, expand_query

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

RRF_K = 60
TOP_K = 5

# Expanded multi-query benchmark dataset covering all modules and safety policies
BENCHMARK_QUERIES = [
    {
        "question": "How does Neuro Mates monitor Parkinson's disease?",
        "expected": {"NM-007", "NM-008", "NM-009", "NM-010"}
    },
    {
        "question": "What hardware and location features are used for Alzheimer's patients?",
        "expected": {"NM-011", "NM-012", "NM-013"}
    },
    {
        "question": "How does the Epilepsy module detect seizures and send alerts?",
        "expected": {"NM-016", "NM-017", "NM-018", "NM-019"}
    },
    {
        "question": "What physiotherapy exercises and sensors are used for rehabilitation?",
        "expected": {"NM-020", "NM-021", "NM-022", "NM-023"}
    },
    {
        "question": "Can Neuro Mates diagnose neurological conditions or replace a doctor?",
        "expected": {"NM-029", "NM-039", "NM-042"}
    }
]


def calculate_weighted_rrf(bm25, dense, bm25_weight=2.0, dense_weight=1.0):
    scores = {}

    for rank, chunk in enumerate(bm25, 1):
        chunk_id = chunk["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + (
            bm25_weight / (RRF_K + rank)
        )

    for rank, chunk in enumerate(dense, 1):
        chunk_id = chunk["id"]
        scores[chunk_id] = scores.get(chunk_id, 0) + (
            dense_weight / (RRF_K + rank)
        )

    return sorted(scores, key=scores.get, reverse=True)


def evaluate_query(results, expected_ids):
    top_results = results[:TOP_K]
    relevant_found = len(set(top_results) & set(expected_ids))
    recall = relevant_found / len(expected_ids)

    first_relevant_rank = next(
        (i for i, chunk_id in enumerate(results, 1) if chunk_id in expected_ids),
        None
    )
    mrr = 1 / first_relevant_rank if first_relevant_rank else 0.0

    return recall, mrr, top_results


if __name__ == "__main__":
    print("=" * 60)
    print("NEURO MATES RETRIEVAL BENCHMARK EVALUATION")
    print("=" * 60)

    # 1. Offline evaluation of weighted RRF on cached Parkinson results
    with open(DATA_DIR / "bm25_results.json", encoding="utf-8") as f:
        bm25_cached = json.load(f)

    with open(DATA_DIR / "dense_results.json", encoding="utf-8") as f:
        dense_cached = json.load(f)["results"]

    print("\n--- Weight Optimization on Cached Query 1 (Parkinson) ---")
    parkinson_expected = BENCHMARK_QUERIES[0]["expected"]

    for bm25_w, dense_w in [(1.0, 1.0), (1.5, 1.0), (2.0, 1.0), (3.0, 1.0)]:
        rrf_ranks = calculate_weighted_rrf(bm25_cached, dense_cached, bm25_w, dense_w)
        recall, mrr, top_5 = evaluate_query(rrf_ranks, parkinson_expected)
        print(f"Weight (BM25: {bm25_w:.1f}, Dense: {dense_w:.1f}) -> Recall@5: {recall:.2f} | MRR: {mrr:.2f} | Top 5: {top_5}")

    # 2. Live BM25 Evaluation across all 5 benchmark questions
    print("\n--- BM25 Evaluation Across Full Benchmark (5 Queries) ---")
    retriever = create_bm25_retriever()
    total_recall = 0.0
    total_mrr = 0.0

    for idx, item in enumerate(BENCHMARK_QUERIES, 1):
        q = item["question"]
        expected = item["expected"]
        expanded = expand_query(q)
        docs = retriever.invoke(expanded)
        retrieved_ids = [doc.metadata["id"] for doc in docs]

        recall, mrr, top_5 = evaluate_query(retrieved_ids, expected)
        total_recall += recall
        total_mrr += mrr

        print(f"\n[Q{idx}] {q}")
        print(f"Expected: {sorted(list(expected))}")
        print(f"Top 5:    {top_5}")
        print(f"Recall@5: {recall:.2f} | MRR: {mrr:.2f}")

    print("\n" + "=" * 60)
    print(f"OVERALL BENCHMARK AVERAGE:")
    print(f"Mean Recall@5: {total_recall / len(BENCHMARK_QUERIES):.2f}")
    print(f"Mean MRR:      {total_mrr / len(BENCHMARK_QUERIES):.2f}")
    print("=" * 60)