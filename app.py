import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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
    
    # Validate environment variables
    required_env_vars = ["GROQ_API_KEY", "DB_HOST", "DB_NAME", "DB_USER", "DB_PASSWORD"]
    missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    print("Initializing Groq client...")
    groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    print("Connecting to database...")
    print(f"DB Host: {os.environ.get('DB_HOST')}")
    print(f"DB Name: {os.environ.get('DB_NAME')}")
    print(f"DB User: {os.environ.get('DB_USER')}")
    
    try:
        db_conn = psycopg2.connect(
            host=os.environ.get("DB_HOST"),
            dbname=os.environ.get("DB_NAME"),
            user=os.environ.get("DB_USER"),
            password=os.environ.get("DB_PASSWORD"),
            port=int(os.environ.get("DB_PORT", 5432)),
            sslmode="require",
            connect_timeout=10
        )
        db_cur = db_conn.cursor()
        print("✅ Database connected successfully!")
    except psycopg2.OperationalError as e:
        print(f"❌ Database connection failed: {e}")
        raise
    
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
@app.get("/", response_class=HTMLResponse)
def root():
    """Serve the chat UI"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Happy Cars - Ask About Any Car</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                max-width: 800px;
                width: 100%;
                padding: 40px;
            }
            .header { text-align: center; margin-bottom: 30px; }
            .header h1 { color: #333; font-size: 2.5em; margin-bottom: 10px; }
            .header .icon { font-size: 3em; margin-bottom: 10px; }
            .header p { color: #666; font-size: 1.1em; }
            .chat-container {
                background: #f8f9fa;
                border-radius: 15px;
                padding: 20px;
                margin-bottom: 20px;
                max-height: 400px;
                overflow-y: auto;
            }
            .message {
                margin-bottom: 15px;
                padding: 15px;
                border-radius: 10px;
                animation: fadeIn 0.3s ease-in;
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .message.user { background: #667eea; color: white; margin-left: 20%; }
            .message.assistant {
                background: white;
                color: #333;
                margin-right: 20%;
                border: 1px solid #e0e0e0;
            }
            .message.error { background: #fee; color: #c00; border: 1px solid #fcc; }
            .sources {
                margin-top: 10px;
                padding-top: 10px;
                border-top: 1px solid #e0e0e0;
                font-size: 0.9em;
                color: #666;
            }
            .source-item {
                margin: 5px 0;
                padding: 5px;
                background: #f0f0f0;
                border-radius: 5px;
                font-size: 0.85em;
            }
            .input-group { margin-bottom: 15px; }
            label {
                display: block;
                margin-bottom: 5px;
                color: #666;
                font-weight: 500;
                font-size: 0.9em;
            }
            input, textarea {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 10px;
                font-size: 1em;
                transition: border-color 0.3s;
            }
            input:focus, textarea:focus {
                outline: none;
                border-color: #667eea;
            }
            textarea {
                resize: vertical;
                min-height: 60px;
                font-family: inherit;
            }
            .button-group { display: flex; gap: 10px; }
            button {
                flex: 1;
                padding: 15px;
                border: none;
                border-radius: 10px;
                font-size: 1em;
                font-weight: 600;
                cursor: pointer;
                transition: all 0.3s;
            }
            button.primary { background: #667eea; color: white; }
            button.primary:hover:not(:disabled) {
                background: #5568d3;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            button.secondary { background: #e0e0e0; color: #666; }
            button.secondary:hover { background: #d0d0d0; }
            button:disabled { opacity: 0.6; cursor: not-allowed; }
            .loading { display: none; text-align: center; padding: 20px; color: #667eea; }
            .loading.active { display: block; }
            .spinner {
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 40px;
                height: 40px;
                animation: spin 1s linear infinite;
                margin: 0 auto 10px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .suggestions {
                display: flex;
                gap: 10px;
                flex-wrap: wrap;
                margin-bottom: 20px;
            }
            .suggestion-chip {
                background: #f0f0f0;
                padding: 8px 15px;
                border-radius: 20px;
                font-size: 0.9em;
                cursor: pointer;
                transition: all 0.3s;
            }
            .suggestion-chip:hover {
                background: #667eea;
                color: white;
            }
            .empty-state { text-align: center; padding: 40px; color: #999; }
            .empty-state .icon { font-size: 3em; margin-bottom: 15px; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="icon">🚗</div>
                <h1>Happy Cars</h1>
                <p>Your AI-powered car expert for India</p>
            </div>

            <div class="suggestions">
                <div class="suggestion-chip" onclick="useSuggestion('What is the engine power of Hyundai Creta?')">
                    Engine specs
                </div>
                <div class="suggestion-chip" onclick="useSuggestion('Compare Maruti Swift and Hyundai i20')">
                    Compare cars
                </div>
                <div class="suggestion-chip" onclick="useSuggestion('What is the mileage of Honda City?')">
                    Fuel efficiency
                </div>
                <div class="suggestion-chip" onclick="useSuggestion('Tell me about Tata Nexon safety features')">
                    Safety features
                </div>
            </div>

            <div class="chat-container" id="chatContainer">
                <div class="empty-state">
                    <div class="icon">💬</div>
                    <p>Ask me anything about cars available in India!</p>
                </div>
            </div>

            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>Thinking...</p>
            </div>

            <div class="input-group">
                <label for="question">Your Question</label>
                <textarea 
                    id="question" 
                    placeholder="e.g., What is the price of Maruti Swift?"
                    onkeypress="handleKeyPress(event)"
                ></textarea>
            </div>

            <div class="input-group">
                <label for="carModel">Filter by Car Model (Optional)</label>
                <input 
                    type="text" 
                    id="carModel" 
                    placeholder="e.g., Hyundai Creta, Honda City"
                    list="carModels"
                >
                <datalist id="carModels">
                    <option value="Hyundai Creta">
                    <option value="Hyundai i20">
                    <option value="Honda City">
                    <option value="Maruti Swift">
                    <option value="Maruti Baleno">
                    <option value="Tata Nexon">
                </datalist>
            </div>

            <div class="button-group">
                <button class="primary" onclick="askQuestion()" id="askBtn">
                    Ask Question
                </button>
                <button class="secondary" onclick="clearChat()">
                    Clear Chat
                </button>
            </div>
        </div>

        <script>
            const API_URL = window.location.origin + '/chat';

            function useSuggestion(text) {
                document.getElementById('question').value = text;
                document.getElementById('question').focus();
            }

            function handleKeyPress(event) {
                if (event.key === 'Enter' && !event.shiftKey) {
                    event.preventDefault();
                    askQuestion();
                }
            }

            function addMessage(content, type, sources = []) {
                const chatContainer = document.getElementById('chatContainer');
                const emptyState = chatContainer.querySelector('.empty-state');
                
                if (emptyState) {
                    emptyState.remove();
                }

                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${type}`;
                
                let messageHTML = `<div>${content}</div>`;
                
                if (sources && sources.length > 0) {
                    messageHTML += '<div class="sources"><strong>Sources:</strong>';
                    sources.forEach(source => {
                        messageHTML += `<div class="source-item">[${source.id}] ${source.source}</div>`;
                    });
                    messageHTML += '</div>';
                }
                
                messageDiv.innerHTML = messageHTML;
                chatContainer.appendChild(messageDiv);
                chatContainer.scrollTop = chatContainer.scrollHeight;
            }

            function clearChat() {
                const chatContainer = document.getElementById('chatContainer');
                chatContainer.innerHTML = `
                    <div class="empty-state">
                        <div class="icon">💬</div>
                        <p>Ask me anything about cars available in India!</p>
                    </div>
                `;
            }

            function showLoading(show) {
                const loading = document.getElementById('loading');
                const askBtn = document.getElementById('askBtn');
                
                if (show) {
                    loading.classList.add('active');
                    askBtn.disabled = true;
                } else {
                    loading.classList.remove('active');
                    askBtn.disabled = false;
                }
            }

            async function askQuestion() {
                const question = document.getElementById('question').value.trim();
                const carModel = document.getElementById('carModel').value.trim();

                if (!question) {
                    alert('Please enter a question!');
                    return;
                }

                addMessage(question, 'user');
                showLoading(true);
                document.getElementById('question').value = '';

                try {
                    const response = await fetch(API_URL, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            question: question,
                            model: carModel || null
                        })
                    });

                    if (!response.ok) {
                        throw new Error(`API error: ${response.status}`);
                    }

                    const data = await response.json();
                    addMessage(data.answer, 'assistant', data.sources);

                } catch (error) {
                    console.error('Error:', error);
                    addMessage(
                        `Sorry, I encountered an error: ${error.message}. Please try again.`,
                        'error'
                    );
                } finally {
                    showLoading(false);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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
