import json

file_path = "data/neuromates_rag_chunks.jsonl"

with open(file_path, "r", encoding="utf-8") as file:
    chunks = [json.loads(line) for line in file if line.strip()]

print("Total chunks:", len(chunks))

print("\nFirst chunk:")
print(json.dumps(chunks[0], indent=4, ensure_ascii=False))

print("\nAvailable categories:")
categories = set(chunk["category"] for chunk in chunks)
for category in sorted(categories):
    print("-", category)
