# Happy Cars RAG API 

A Retrieval-Augmented Generation (RAG) system for automotive sales assistance, specifically designed for the Indian car market. This API provides intelligent responses about car specifications, features, and pricing using official documentation, Wikipedia, and automotive datasets.

## Features

- 🔍 **Intelligent Search**: Vector similarity search with pgvector
- 🎯 **Reranking**: Cross-encoder reranking for improved relevance
- 📊 **Multi-source Data**: Combines official specs, Wikipedia, and structured datasets
- 🤖 **LLM-powered Answers**: Uses Groq's Llama 3.1 for natural responses
- ✅ **Citation Support**: All answers include source citations
- 🚀 **Production Ready**: FastAPI with proper error handling and health checks

## Architecture

```
User Query → Embedding → Vector Search (PostgreSQL + pgvector) 
→ Candidate Retrieval → Cross-Encoder Reranking → Context Building 
→ LLM Generation (Groq) → Response with Citations
```

## Tech Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL with pgvector extension (Supabase)
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Reranking**: cross-encoder/ms-marco-MiniLM-L-6-v2
- **LLM**: Groq (Llama 3.1 8B Instant)

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL with pgvector extension
- Groq API key

### Local Development

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd happy-cars-rag
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your actual credentials
```

5. **Run the application**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

6. **Access the API**
- API: http://localhost:8000
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## API Endpoints

### POST /chat

Ask questions about cars.

**Request Body:**
```json
{
  "question": "What is the engine power of Hyundai Creta?",
  "model": "Hyundai Creta"  // Optional: filter by specific car model
}
```

**Response:**
```json
{
  "answer": "The maximum power of the Hyundai Creta is 113.45 BHP.\n\nSources:\n[1] India New Cars Dataset (Cardekho-derived, 2022)",
  "sources": [
    {
      "id": 1,
      "source": "India New Cars Dataset (Cardekho-derived, 2022)"
    }
  ]
}
```

### GET /health

Health check endpoint to verify API and database connectivity.

**Response:**
```json
{
  "status": "healthy",
  "database": "connected",
  "models_loaded": true
}
```

### GET /

Root endpoint with API information.

## Data Sources

The system uses three types of data:

1. **Official Specifications**: Scraped from manufacturer websites
   - Hyundai i20, Creta
   - Honda City
   - Maruti Swift, Baleno
   - Tata Nexon

2. **Wikipedia**: Comprehensive car information and history

3. **Structured Dataset**: 2022 Indian car specifications
   - 203 car models
   - Engine specs, pricing, ratings

## Database Schema

```sql
CREATE TABLE car_documents (
    id SERIAL PRIMARY KEY,
    model VARCHAR(255),
    doc_type VARCHAR(100),
    content TEXT,
    source TEXT,
    embedding vector(384)  -- pgvector extension
);

CREATE INDEX ON car_documents 
USING ivfflat (embedding vector_cosine_ops);
```

## Deployment on Render

### Step 1: Prepare Your Repository

1. Create a GitHub repository with the following structure:
```
happy-cars-rag/
├── app.py
├── requirements.txt
├── .env.example
├── README.md
└── .gitignore
```

2. Create `.gitignore`:
```
.env
__pycache__/
*.pyc
venv/
.DS_Store
```

3. Push to GitHub:
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-github-repo-url>
git push -u origin main
```

### Step 2: Deploy on Render

1. **Sign up/Login** to [Render](https://render.com)

2. **Create New Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository

3. **Configure the service**:
   - **Name**: `happy-cars-rag-api`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`
   - **Instance Type**: Choose based on your needs (Free tier available)

4. **Add Environment Variables**:
   - `DB_HOST`: Your Supabase host
   - `DB_NAME`: postgres
   - `DB_USER`: Your database user
   - `DB_PASSWORD`: Your database password
   - `DB_PORT`: 5432
   - `GROQ_API_KEY`: Your Groq API key

5. **Deploy**: Click "Create Web Service"

### Step 3: Verify Deployment

Once deployed, your API will be available at:
```
https://happy-cars-rag-api.onrender.com
```

Test it:
```bash
curl -X POST https://happy-cars-rag-api.onrender.com/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the mileage of Maruti Swift?"}'
```

## Example Usage

### Python
```python
import requests

url = "http://localhost:8000/chat"
payload = {
    "question": "Compare the engine power of Hyundai Creta and Honda City",
    "model": None  # Search across all models
}

response = requests.post(url, json=payload)
print(response.json())
```

### cURL
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the safety features of Maruti Baleno?",
    "model": "Maruti Baleno"
  }'
```

### JavaScript
```javascript
const response = await fetch('http://localhost:8000/chat', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    question: 'What is the price range of Tata Nexon?',
    model: 'Tata Nexon'
  })
});

const data = await response.json();
console.log(data);
```

## Performance Optimization

- **Vector Search**: Uses pgvector's IVFFlat index for fast similarity search
- **Reranking**: Limited to top 12 candidates to balance speed and accuracy
- **Model Loading**: Models loaded once at startup and kept in memory
- **Connection Pooling**: Database connection reused across requests

## Monitoring

- Check health: `GET /health`
- Logs available in Render dashboard
- Monitor response times and error rates

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License.

## Support

For questions or issues:
- Open an issue on GitHub
- Contact: [balajikamaraj99@gmail.com]

## Acknowledgments

- Car data sourced from official manufacturer websites
- Wikipedia for comprehensive car information
- Cardekho dataset for structured specifications
- Groq for fast LLM inference
- Supabase for managed PostgreSQL with pgvector

---

**Built with ❤️ for Indian car buyers**
