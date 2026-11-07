import json
from pathlib import Path

INPUT_FILE = Path("data/neuromates_rag_chunks.jsonl")

TARGET_IDS = {
    "NM-001",
    "NM-007",
    "NM-008",
    "NM-010",
    "NM-031",
    "NM-032",
    "NM-035",
}

with open(INPUT_FILE, "r", encoding="utf-8") as file:

    for line in file:
        if not line.strip():
            continue

        chunk = json.loads(line)

        if chunk["id"] in TARGET_IDS:

            print("\n" + "=" * 70)
            print("ID:", chunk["id"])
            print("TOPIC:", chunk.get("topic"))
            print("-" * 70)
            print(chunk["content"])