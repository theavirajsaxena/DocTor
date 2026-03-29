# DocTor — Web-Scale RAG System

A web-based Retrieval-Augmented Generation (RAG) system for technical document understanding.

## Features
- Adaptive semantic chunking
- Dense-Sparse hybrid indexing (FAISS + BM25)
- Reciprocal Rank Fusion (RRF) retrieval
- Evidence-aligned answer generation with inline citations
- Clean chat UI with expandable citation cards
- Response caching for repeated queries

## Tech Stack
- **Backend:** Python, FastAPI, Uvicorn
- **NLP:** spaCy, sentence-transformers (all-MiniLM-L6-v2)
- **Retrieval:** FAISS (dense) + BM25 (sparse) + RRF fusion
- **LLM:** GLM-4.5-Air via OpenRouter
- **Frontend:** HTML, CSS, Vanilla JavaScript

## Setup

### 1. Clone the repository
\git clone https://github.com/YOUR_USERNAME/doctor-rag.git
cd doctor-rag
\
### 2. Create virtual environment
\py -3.11 -m venv venv
venv\Scripts\activate
\
### 3. Install dependencies
\pip install -r requirements.txt
python -m spacy download en_core_web_sm
\
### 4. Set up environment variables
Create a \.env\ file in the root directory:
\OPENROUTER_API_KEY=your_openrouter_key_here
\
### 5. Run the server
\uvicorn main:app --reload
\
### 6. Open the UI
\http://127.0.0.1:8000/static/index.html
\
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
\doctor-rag/
├── main.py          # FastAPI app and all endpoints
├── extractor.py     # PDF text extraction
├── chunker.py       # Adaptive semantic chunking
├── indexer.py       # FAISS + BM25 hybrid indexing
├── retriever.py     # RRF retrieval with re-ranking
├── generator.py     # LLM answer generation
├── test_doctor.py   # Automated test suite
├── static/
│   └── index.html   # Frontend chat UI
├── uploads/         # Uploaded PDFs (gitignored)
└── requirements.txt
\
## B.Tech Major Project
Department of Computer Science & Engineering
