import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import psycopg2
from sentence_transformers import SentenceTransformer, CrossEncoder
from groq import Groq
from contextlib import asynccontextmanager

# Global variables for models and connections
embedder = None
reranker = None
groq_client = None
db_conn = None
db_cur = None

def get_embedder():
    """Lazy load embedder only when needed"""
    global embedder
    if embedder is None:
        print("Loading embedder...")
        embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return embedder

def get_reranker():
    """Lazy load reranker only when needed"""
    global reranker
    if reranker is None:
        print("Loading reranker...")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return reranker

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Initialize only essential connections
    global groq_client, db_conn, db_cur
    
    print("Initializing Groq client...")
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    print("Connecting to database...")
    db_conn = psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        dbname=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        port=int(os.environ.get("DB_PORT", 5432)),
        sslmode="require"
    )
    db_cur = db_conn.cursor()
    
    print("Startup complete! Models will load on first request.")
    
    yield
    
    # Shutdown: Close connections
    if db_cur:
        db_cur.close()
    if db_conn:
        db_conn.close()
    print("Shutdown complete!")

app = FastAPI(
    title="Happy Cars RAG API",
    version="1.0",
    description="Automotive sales assistant API for Happy Cars India",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    model: Optional[str] = None

class QueryResponse(BaseModel):
    answer: str
    sources: list

# System prompt
SYSTEM_PROMPT = """
You are an automotive sales assistant for Happy Cars (India).

You MUST answer the user question using ONLY the information provided
in the CONTEXT section below.

Rules (MANDATORY):
1. Do NOT use any outside knowledge.
2. Do NOT guess or infer missing information.
3. If the answer is not explicitly available in the context, reply:
   "The information is not available in the provided data."
4. Use clear, concise, customer-friendly language.
5. Always include citations for every factual statement.
6. Do NOT mention internal systems, embeddings, or databases.

Citation rules:
- Each citation must correspond to a source listed at the end.
- If multiple sources support a statement, cite all.
- Do NOT include a "Sources:" section.
- Use citation numbers like [1], [2] in the answer text only.

Output format:
- Answer in plain text (not markdown).
- Then include a "Sources:" section listing citations.

You are answering for Indian customers.
"""

def retrieve_candidates(query: str, model_filter: Optional[str] = None, limit: int = 12):
    """Retrieve candidate documents from the database"""
    emb = get_embedder()  # Lazy load
    q_embedding = emb.encode(query).tolist()
    
    if model_filter:
        sql = """
        SELECT model, doc_type, content, source
        FROM car_documents
        WHERE model ILIKE %s
        ORDER BY
          CASE doc_type
            WHEN 'official_specs' THEN 1
            WHEN 'new_car_specs_dataset_2022' THEN 2
            WHEN 'wikipedia' THEN 3
            ELSE 4
          END,
          embedding <-> %s::vector
        LIMIT %s;
        """
        db_cur.execute(sql, (f"%{model_filter}%", q_embedding, limit))
    else:
        sql = """
        SELECT model, doc_type, content, source
        FROM car_documents
        ORDER BY embedding <-> %s::vector
        LIMIT %s;
        """
        db_cur.execute(sql, (q_embedding, limit))
    
    rows = db_cur.fetchall()
    
    return [
        {
            "model": r[0],
            "type": r[1],
            "text": r[2],
            "source": r[3]
        }
        for r in rows
    ]

def is_good_spec_chunk(text: str) -> bool:
    """Filter out noisy official spec chunks"""
    text_lower = text.lower()
    
    spec_indicators = ["bhp", "nm", "cc", "engine", "power", "torque"]
    ui_noise = ["otp", "menu", "share", "submit", "click", "stay in touch"]
    
    if not any(s in text_lower for s in spec_indicators):
        return False
    
    if any(n in text_lower for n in ui_noise):
        return False
    
    return True

def rerank_docs(query: str, docs: list, top_k: int = 3):
    """Rerank documents using cross-encoder"""
    filtered = []
    
    for d in docs:
        if d["type"] == "official_specs":
            if is_good_spec_chunk(d["text"]):
                filtered.append(d)
        else:
            filtered.append(d)
    
    if not filtered:
        return []
    
    ranker = get_reranker()  # Lazy load
    pairs = [(query, d["text"]) for d in filtered]
    scores = ranker.predict(pairs)
    
    ranked = sorted(
        zip(scores, filtered),
        key=lambda x: x[0],
        reverse=True
    )
    
    return [doc for _, doc in ranked[:top_k]]

def build_context(docs: list) -> str:
    """Build context string from documents"""
    blocks = []
    
    for i, d in enumerate(docs, start=1):
        block = (
            f"[{i}] Source: {d['source']}\n"
            f"Content:\n{d['text']}"
        )
        blocks.append(block)
    
    return "\n\n".join(blocks)

def answer_query(query: str, model_filter: Optional[str] = None) -> dict:
    """Main RAG pipeline"""
    # 1. Retrieve candidates
    candidates = retrieve_candidates(query, model_filter=model_filter)
    
    if not candidates:
        return {
            "answer": "The information is not available in the provided data.",
            "sources": []
        }
    
    # 2. Rerank
    top_docs = rerank_docs(query, candidates)
    
    if not top_docs:
        return {
            "answer": "The information is not available in the provided data.",
            "sources": []
        }
    
    # 3. Build context
    context = build_context(top_docs)
    
    user_prompt = f"""
CONTEXT:
{context}

QUESTION:
{query}

ANSWER:
"""
    
    # 4. Call LLM (Groq)
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            max_tokens=512
        )
        
        answer = response.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")
    
    sources = [
        {"id": i + 1, "source": d["source"]}
        for i, d in enumerate(top_docs)
    ]
    
    return {
        "answer": answer,
        "sources": sources
    }

# API Endpoints
@app.get("/")
def root():
    return {
        "message": "Happy Cars RAG API",
        "version": "1.0",
        "endpoints": {
            "POST /chat": "Ask questions about cars",
            "GET /health": "Health check"
        }
    }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        # Test database connection
        db_cur.execute("SELECT 1;")
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "database": db_status,
        "models": {
            "embedder": "loaded" if embedder is not None else "not_loaded",
            "reranker": "loaded" if reranker is not None else "not_loaded"
        }
    }

@app.post("/chat", response_model=QueryResponse)
def chat(req: QueryRequest):
    """
    Main chat endpoint for asking questions about cars
    
    - **question**: The user's question
    - **model**: Optional filter for specific car model (e.g., "Hyundai Creta")
    """
    try:
        result = answer_query(
            query=req.question,
            model_filter=req.model
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
