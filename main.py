
from hybrid_search import (
    create_bm25_retriever,
    hybrid_search
)

from answer_generator import generate_answer


def main():

    print("=" * 60)
    print("NEURO MATES RAG SYSTEM")
    print("=" * 60)

    # 1. Build BM25 index
    bm25_retriever = create_bm25_retriever()

    print("\nBM25 index created!")
    print("Neuro Mates RAG is ready!")

    # 2. Get user question
    question = input("\nAsk Neuro Mates something: ").strip()

    if not question:
        print("Please enter a question.")
        return

    # 3. Retrieve relevant chunks
    retrieved_chunks = hybrid_search(
        question,
        bm25_retriever
    )

    if not retrieved_chunks:
        print("\nNo relevant information found.")
        return

    # 4. Generate final answer
    print("\nGenerating answer using Gemini...")

    answer = generate_answer(
        question,
        retrieved_chunks
    )

    # 5. Display final response
    print("\n" + "=" * 60)
    print("NEURO MATES ANSWER")
    print("=" * 60)

    print(answer)


if __name__ == "__main__":
    main()
