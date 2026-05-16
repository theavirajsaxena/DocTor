# DocTor - Web-Scale RAG System

A web-based Retrieval-Augmented Generation (RAG) system for technical document understanding.

## Features
- Adaptive semantic chunking
- Dense-sparse hybrid indexing with FAISS and BM25
- Reciprocal Rank Fusion retrieval
- Evidence-aligned answer generation with inline citations
- Clean chat UI with expandable citation cards
- Per-browser sessions and response caching
- Upload/query limits for safer deployment

## Tech Stack
- Backend: Python, FastAPI, Uvicorn
- NLP: spaCy, sentence-transformers (all-MiniLM-L6-v2)
- Retrieval: FAISS + BM25 + RRF fusion
- LLM: GLM-4.5-Air via OpenRouter by default
- Frontend: HTML, CSS, Vanilla JavaScript

## Local Setup

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create a `.env` file:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
CORS_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
```

Preload local models, then run the API:

```powershell
python scripts/preload_models.py
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000/static/index.html`.

## Deployment

Required environment variables:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
CORS_ORIGINS=https://your-deployed-domain.com
```

Optional environment variables:

```env
OPENROUTER_MODEL=z-ai/glm-4.5-air:free
MAX_UPLOAD_MB=25
MAX_PDF_PAGES=200
MAX_QUERY_CHARS=1000
MAX_TOP_K=10
MAX_SESSIONS=100
SESSION_TTL_MINUTES=120
MODEL_CACHE_DIR=model_cache
```

Recommended build command:

```powershell
pip install -r requirements.txt
python scripts/preload_models.py
```

Recommended start command:

```powershell
python -m uvicorn main:app --host 0.0.0.0 --port $env:PORT
```

For Linux-based hosts, use `$PORT` instead of `$env:PORT`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | /upload | Upload a PDF document |
| POST | /chunk | Adaptively chunk the document |
| POST | /index | Build hybrid FAISS + BM25 index |
| POST | /retrieve | Retrieve relevant chunks for a query |
| POST | /ask | Generate a grounded answer |
| GET | /citations | Get cited passages from last answer |
| GET | /history | Get full conversation history |
| GET | /stats | Get system performance statistics |
| GET | /test/health | Full system health check |
| POST | /reset | Reset the current session |

## Project Structure

```text
doctor-rag/
|-- main.py
|-- extractor.py
|-- chunker.py
|-- indexer.py
|-- retriever.py
|-- generator.py
|-- test_doctor.py
|-- scripts/
|   `-- preload_models.py
|-- static/
|   `-- index.html
|-- uploads/
|-- requirements.txt
|-- Procfile
`-- runtime.txt
```

## Testing

Start the server first, then run:

```powershell
python test_doctor.py
```

The script expects a live server at `127.0.0.1:8000` and at least one sample PDF in `uploads/`.
