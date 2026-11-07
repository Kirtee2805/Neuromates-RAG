import time
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from hybrid_search import get_bm25_retriever, hybrid_search
from answer_generator import generate_answer

app = FastAPI(
    title="Neuro Mates RAG API",
    description="Care-support knowledge retrieval and QA API for Neuro Mates",
    version="1.0.0"
)

# Enable CORS for frontend applications (web dashboard / portals)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Warm up in-memory index on startup
retriever = get_bm25_retriever()


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, example="How does Neuro Mates monitor Parkinson's disease?")
    top_k: Optional[int] = Field(default=5, ge=1, le=10)


class SearchResponse(BaseModel):
    question: str
    chunks: List[dict]
    count: int


class ChatResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    latency_seconds: float


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "neuromates-rag",
        "indexed_chunks": 45
    }


@app.post("/api/search", response_model=SearchResponse)
def search_knowledge(request: QueryRequest):
    chunks = hybrid_search(request.question, bm25_retriever=retriever, top_k=request.top_k)
    return SearchResponse(
        question=request.question,
        chunks=chunks,
        count=len(chunks)
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat_knowledge(request: QueryRequest):
    start = time.time()
    chunks = hybrid_search(request.question, bm25_retriever=retriever, top_k=request.top_k)
    
    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant knowledge chunks found.")

    answer = generate_answer(request.question, chunks)
    sources = [c["id"] for c in chunks[:4]]

    return ChatResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        latency_seconds=round(time.time() - start, 2)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

