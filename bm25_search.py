import json
from pathlib import Path

from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

BASE_DIR = Path(__file__).resolve().parent

INPUT_FILE = BASE_DIR / "data" / "neuromates_rag_chunks.jsonl"

TOP_K = 10


def load_documents():

    documents = []

    with open(INPUT_FILE, "r", encoding="utf-8") as file:

        for line in file:

            if not line.strip():
                continue

            chunk = json.loads(line)

            document = Document(
                page_content=(
                    f"Topic: {chunk.get('topic', '')}\n"
                    f"Content: {chunk['content']}"
                ),
                metadata={
                    key: value
                    for key, value in chunk.items()
                    if key != "content"
                }
            )

            documents.append(document)

    return documents

def create_bm25_retriever():

    documents = load_documents()

    print("Total documents:", len(documents))

    retriever = BM25Retriever.from_documents(
        documents,
        k=TOP_K
    )

    print("BM25 index created successfully!")

    return retriever

def search_bm25(question, retriever):

    results = retriever.invoke(question)

    print("\nBM25 Search Results")
    print("=" * 60)

    for rank, document in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print(f"Chunk ID: {document.metadata.get('id')}")
        print(f"Topic: {document.metadata.get('topic')}")
        print(f"Content: {document.page_content}")

        print("-" * 60)


# 4. Main

if __name__ == "__main__":

    retriever = create_bm25_retriever()

    question = input(
        "\nAsk Neuro Mates something: "
    ).strip()

    if question:
        search_bm25(question, retriever)
    else:   
        print("Please enter a question.")