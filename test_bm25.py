import json
from hybrid_search import (
    create_bm25_retriever,
    expand_query
)

retriever = create_bm25_retriever()

question = "How does Neuro Mates monitor Parkinson's disease?"

expanded_question = expand_query(question)

print("Expanded query:")
print(expanded_question)

results = retriever.invoke(expanded_question)

print("\nBM25 RESULTS")

for rank, document in enumerate(results, start=1):
    print(
        rank,
        document.metadata["id"],
        document.metadata.get("topic")
    )

with open("data/bm25_results.json", "w", encoding="utf-8") as file:
    json.dump(
        [
            {
                "id": doc.metadata["id"],
                "topic": doc.metadata.get("topic", ""),
                "content": doc.page_content
            }
            for doc in results
        ],
        file,
        ensure_ascii=False,
        indent=2
    )

print("\nBM25 results saved!")