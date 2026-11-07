
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

llm = ChatGoogleGenerativeAI(
    model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
    google_api_key=api_key,
    max_retries=3,
    request_timeout=30.0
)

import time

if __name__ == "__main__":
    print("Sending request to Gemini...")

    start = time.time()

    response = llm.invoke(
        "Reply with exactly: Gemini connected."
    )

    end = time.time()

    print("Response:", response.content)
    print(f"Time taken: {end - start:.2f} seconds")
