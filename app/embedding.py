import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set in the environment variables.")

embeddings = GoogleGenerativeAIEmbeddings(
    model = "gemini-embedding-2-preview",
    google_api_key = API_KEY,
    output_dimensionality = 3072
)


print ("Embedding model :",embeddings.model)