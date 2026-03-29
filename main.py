# main.py
import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from extractor import extract_text_from_pdf
from chunker import adaptive_chunk
from indexer import build_index
from retriever import retrieve
from generator import generate_answer
import hashlib

def _cache_key(query: str) -> str:
    """Creates a short hash key for a query string."""
    return hashlib.md5(query.strip().lower().encode()).hexdigest()

load_dotenv()  # loads your GEMINI_API_KEY from .env

app = FastAPI(title="DocTor RAG System")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ### 7.3 — Open the UI

# Make sure your server is running, then open your browser and go to:
# ```
# http://127.0.0.1:8000/static/index.html

# Serve the frontend HTML from the /static folder
app.mount("/static", StaticFiles(directory="static"), name="static")

from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")

# In-memory store for the current document session
# (In a real app you'd use a database, but this is fine for your project)
document_store = {
    "filename": None,
    "pages": [],       # raw extracted pages
    "chunks": [],      # will be filled in Step 3
    "index": None,      # will be filled in Step 4
    "last_answer": {},
    "history"    : [],
    "cache"      : {}
}

UPLOAD_DIR = "uploads"


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Accepts a PDF upload, saves it, extracts text from it.
    """
    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save the uploaded file to disk
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Extract text page by page
    pages = extract_text_from_pdf(file_path)

    if not pages:
        raise HTTPException(status_code=400, detail="Could not extract text. Is the PDF scanned/image-based?")

    # Store in session
    document_store["filename"] = file.filename
    document_store["pages"] = pages
    document_store["chunks"] = []
    document_store["index"] = None

    return JSONResponse({
        "message": "Document uploaded and text extracted successfully.",
        "filename": file.filename,
        "total_pages": len(pages),
        "preview": pages[0]["text"][:300] + "..."  # first 300 chars as preview
    })

@app.get("/document/info")
def document_info():
    """
    Returns info about the currently loaded document.
    """
    if not document_store["filename"]:
        raise HTTPException(status_code=404, detail="No document uploaded yet.")

    return {
        "filename": document_store["filename"],
        "total_pages": len(document_store["pages"]),
        "total_chunks": len(document_store["chunks"]),
        "index_ready": document_store["index"] is not None
    }

@app.post("/chunk")
def chunk_document():
    """
    Chunks the currently loaded document using adaptive chunking.
    Must call /upload first.
    """
    if not document_store["pages"]:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload a PDF first.")

    chunks = adaptive_chunk(document_store["pages"])
    document_store["chunks"] = chunks

    # Build a small readable summary for the response
    word_counts = [c["word_count"] for c in chunks]
    avg_words   = sum(word_counts) // len(word_counts) if word_counts else 0

    return {
        "message"       : "Document chunked successfully.",
        "total_chunks"  : len(chunks),
        "avg_chunk_words": avg_words,
        "min_chunk_words": min(word_counts) if word_counts else 0,
        "max_chunk_words": max(word_counts) if word_counts else 0,
        "sample_chunk"  : chunks[0] if chunks else None
    }
@app.post("/index")
def index_document():
    """
    Builds FAISS + BM25 index from the chunked document.
    Must call /chunk first.
    """
    if not document_store["chunks"]:
        raise HTTPException(
            status_code=400,
            detail="No chunks found. Please run /chunk first."
        )

    index_data = build_index(document_store["chunks"])
    document_store["index"] = index_data

    return {
        "message"      : "Hybrid index built successfully.",
        "total_vectors": len(document_store["chunks"]),
        "index_type"   : "FAISS (dense) + BM25 (sparse) with RRF fusion"
    }


@app.post("/retrieve")
def retrieve_chunks(query: str, top_k: int = 5):
    """
    Retrieves the most relevant chunks for a given query.
    Must call /index first.
    """
    if document_store["index"] is None:
        raise HTTPException(
            status_code=400,
            detail="Index not built yet. Please run /index first."
        )

    results = retrieve(query, document_store["index"], top_k)

    return {
        "query"          : query,
        "top_k"          : top_k,
        "retrieved_chunks": results
    }
@app.post("/ask")
def ask_question(query: str, top_k: int = 5):
    """
    Full RAG pipeline with caching, re-ranking and deduplication.
    """
    if document_store["index"] is None:
        raise HTTPException(
            status_code=400,
            detail="Index not ready. Please upload, chunk and index first."
        )

    # Check cache first
    key = _cache_key(query)
    if key in document_store["cache"]:
        cached = document_store["cache"][key]
        return {
            "query"      : query,
            "answer"     : cached["answer"],
            "citations"  : cached["citations"],
            "chunks_used": cached["chunks_used"],
            "cached"     : True
        }

    # Retrieve with optimized pipeline
    retrieved = retrieve(query, document_store["index"], top_k)

    # Generate answer
    result = generate_answer(query, retrieved)

    # Store in cache and history
    cache_entry = {
        "answer"    : result["answer"],
        "citations" : result["citations"],
        "chunks_used": len(retrieved)
    }
    document_store["cache"][key]  = cache_entry
    document_store["last_answer"] = result
    document_store["history"].append({
        "query"    : query,
        "answer"   : result["answer"],
        "citations": result["citations"],
        "timestamp": __import__("datetime").datetime.now().isoformat()
    })

    return {
        "query"      : query,
        "answer"     : result["answer"],
        "citations"  : result["citations"],
        "chunks_used": len(retrieved),
        "cached"     : False
    }


@app.get("/history")
def get_history():
    """Returns the full conversation history for the session."""
    return {
        "total_questions": len(document_store["history"]),
        "history"        : document_store["history"]
    }


@app.post("/reset")
def reset_session():
    document_store["filename"]    = None
    document_store["pages"]       = []
    document_store["chunks"]      = []
    document_store["index"]       = None
    document_store["last_answer"] = {}
    document_store["history"]     = []
    document_store["cache"]       = {}
    return {"message": "Session reset successfully."}
@app.get("/citations")
def get_citations():
    """
    Returns the full cited passages from the last /ask call.
    The frontend uses this to highlight source text.
    """
    if "last_answer" not in document_store:
        raise HTTPException(
            status_code=404,
            detail="No answer generated yet. Call /ask first."
        )

    result = document_store["last_answer"]

    return {
        "total_citations" : len(result["citations"]),
        "citations"       : result["citations"]
    }
@app.get("/stats")
def get_stats():
    """
    Returns system performance statistics for the current session.
    Useful for project evaluation and reporting.
    """
    chunks = document_store["chunks"]
    history = document_store["history"]

    if not chunks:
        return {"message": "No document loaded yet."}

    word_counts = [c["word_count"] for c in chunks]

    return {
        "document": {
            "filename"       : document_store["filename"],
            "total_pages"    : len(document_store["pages"]),
            "total_chunks"   : len(chunks),
            "avg_chunk_words": round(sum(word_counts)/len(word_counts), 1),
            "min_chunk_words": min(word_counts),
            "max_chunk_words": max(word_counts),
        },
        "session": {
            "questions_asked": len(history),
            "cache_hits"     : len(document_store["cache"]),
            "cache_size"     : len(document_store["cache"]),
        },
        "system": {
            "retrieval_method": "Dense (FAISS) + Sparse (BM25) + RRF Fusion",
            "chunking_method" : "Adaptive semantic boundary detection",
            "llm_model"       : "GLM-4.5-Air via OpenRouter",
            "embedding_model" : "all-MiniLM-L6-v2 (384 dims)"
        }
    }
@app.get("/test/health")
def health_check():
    """
    Full system health check.
    Tests every component and reports status.
    """
    results = {}

    # Check document loaded
    results["document_loaded"] = {
        "status" : "pass" if document_store["filename"] else "fail",
        "detail" : document_store["filename"] or "No document uploaded"
    }

    # Check chunks exist
    results["chunking"] = {
        "status": "pass" if document_store["chunks"] else "fail",
        "detail": f"{len(document_store['chunks'])} chunks" 
                  if document_store["chunks"] else "Not chunked yet"
    }

    # Check index built
    results["index"] = {
        "status": "pass" if document_store["index"] else "fail",
        "detail": "FAISS + BM25 ready" 
                  if document_store["index"] else "Not indexed yet"
    }

    # Check cache
    results["cache"] = {
        "status": "pass",
        "detail": f"{len(document_store['cache'])} queries cached"
    }

    # Check history
    results["history"] = {
        "status": "pass",
        "detail": f"{len(document_store['history'])} questions answered"
    }

    # Overall status
    all_pass = all(
        v["status"] == "pass" 
        for v in results.values()
    )

    return {
        "overall" : "healthy" if all_pass else "degraded",
        "checks"  : results
    }
# ```

# ---

# ### 5.3 — Verify Your .env File

# Open your `.env` file and make sure it looks exactly like this — no spaces around the `=` sign:
# ```
# GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXX
# # ```

# ---

# ### 4.4 — Test the Full Pipeline

# Restart the server if needed, then go to `http://127.0.0.1:8000/docs` and run these **in order**:

# **1.** `POST /upload` — upload your PDF

# **2.** `POST /chunk` — chunk it

# **3.** `POST /index` — build the hybrid index (this may take 20–30 seconds the first time as it downloads the MiniLM embedding model ~80MB)

# **4.** `POST /retrieve` — test a query. In the query box type something relevant to your PDF, for example:
# ```
# What is the main methodology used?
# # ```

# ---

# ### 3.3 — Test It

# Make sure uvicorn is still running (if not, run `uvicorn main:app --reload` again). Go to:
# ```
# http://127.0.0.1:8000/docs
# # ```

# **What this does:** FastAPI creates two endpoints — `/upload` accepts your PDF and immediately extracts text from it, and `/document/info` lets you check what's currently loaded. The `document_store` dictionary acts as your session memory for now.

# ---

# ### 2.3 — Run the Server

# In your Command Prompt (with venv active, inside `doctor_rag`), run:
# ```
# uvicorn main:app --reload
# ```

# You should see output like:
# ```
# INFO:     Uvicorn running on http://127.0.0.1:8000
# INFO:     Application startup complete.
# ```

# The `--reload` flag means the server automatically restarts whenever you save changes to your code — very handy during development.

# ---

# ### 2.4 — Test It

# Open your browser and go to:
# ```
# http://127.0.0.1:8000
# ```

# You should see: `{"message": "DocTor RAG API is running"}`

# Now test the upload using FastAPI's **built-in interactive docs** — go to:
# ```
# http://127.0.0.1:8000/docs