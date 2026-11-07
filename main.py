from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from hybrid_search import create_bm25_retriever,hybrid_search
from answer_generator import generate_answer

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

bm25_retriever = create_bm25_retriever()
print("Neuro Mates RAG System Ready on Render!")

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
def ask_neuro_mates(payload: ChatRequest):
    question = payload.question.strip()

    if not question:
        return {"error": "Please enter a question."}

   
    retrieved_chunks = hybrid_search(question, bm25_retriever)

    if not retrieved_chunks:
        return {"reply": "No relevant information found.", "sources": []}

    answer = generate_answer(question, retrieved_chunks)

   
    return {
        "reply": answer
    }


@app.get("/")
def home():
    return {"status": "Neuro Mates RAG API is online and running!"}
