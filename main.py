import hashlib
import os
import re
import uuid
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from chunker import adaptive_chunk
from extractor import extract_text_from_pdf
from generator import generate_answer
from indexer import build_index
from retriever import retrieve


load_dotenv()

UPLOAD_DIR = "uploads"
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "25"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "200"))
MAX_QUERY_CHARS = int(os.getenv("MAX_QUERY_CHARS", "1000"))
MAX_TOP_K = int(os.getenv("MAX_TOP_K", "10"))
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "100"))
SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", "120"))
SESSION_COOKIE_NAME = "doctor_session"
DEFAULT_SESSION_ID = "default"

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="DocTor RAG System")


def _parse_cors_origins() -> list[str]:
    origins = os.getenv("CORS_ORIGINS", "")
    if not origins:
        return ["http://127.0.0.1:8000", "http://localhost:8000"]
    return [origin.strip() for origin in origins.split(",") if origin.strip()]


cors_origins = _parse_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-Session-Id"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


def _new_document_store() -> dict:
    return {
        "filename": None,
        "stored_path": None,
        "pages": [],
        "chunks": [],
        "index": None,
        "last_answer": {},
        "history": [],
        "cache": {},
        "last_seen": datetime.utcnow(),
    }


document_sessions: dict[str, dict] = {DEFAULT_SESSION_ID: _new_document_store()}


def _cache_key(query: str) -> str:
    """Creates a stable hash key for a normalized query string."""
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()


def _session_id_from_request(request: Request) -> str | None:
    session_id = request.headers.get("x-session-id") or request.cookies.get(SESSION_COOKIE_NAME)
    if not session_id:
        return None
    return re.sub(r"[^a-zA-Z0-9_-]", "", session_id)[:80] or None


def _get_document_store(request: Request, response: Response | None = None) -> dict:
    _cleanup_expired_sessions()

    session_id = _session_id_from_request(request)
    if not session_id:
        session_id = str(uuid.uuid4())
        if response is not None:
            response.set_cookie(
                SESSION_COOKIE_NAME,
                session_id,
                httponly=True,
                samesite="lax",
                max_age=SESSION_TTL_MINUTES * 60,
            )

    if session_id not in document_sessions:
        _trim_session_count()
        document_sessions[session_id] = _new_document_store()
    document_sessions[session_id]["last_seen"] = datetime.utcnow()
    return document_sessions[session_id]


def _delete_session_file(document_store: dict) -> None:
    stored_path = document_store.get("stored_path")
    if stored_path and os.path.exists(stored_path):
        os.remove(stored_path)


def _cleanup_expired_sessions() -> None:
    cutoff = datetime.utcnow() - timedelta(minutes=SESSION_TTL_MINUTES)
    expired = [
        session_id
        for session_id, store in document_sessions.items()
        if session_id != DEFAULT_SESSION_ID and store.get("last_seen", datetime.utcnow()) < cutoff
    ]
    for session_id in expired:
        _delete_session_file(document_sessions[session_id])
        del document_sessions[session_id]


def _trim_session_count() -> None:
    if len(document_sessions) < MAX_SESSIONS:
        return

    candidates = [
        (session_id, store.get("last_seen", datetime.utcnow()))
        for session_id, store in document_sessions.items()
        if session_id != DEFAULT_SESSION_ID
    ]
    candidates.sort(key=lambda item: item[1])

    while len(document_sessions) >= MAX_SESSIONS and candidates:
        session_id, _ = candidates.pop(0)
        _delete_session_file(document_sessions[session_id])
        del document_sessions[session_id]


def _validate_pdf_name(filename: str | None) -> str:
    if not filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")
    safe_name = os.path.basename(filename)
    if not safe_name.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    return safe_name


async def _save_limited_upload(file: UploadFile, safe_name: str) -> str:
    stored_name = f"{uuid.uuid4().hex}.pdf"
    file_path = os.path.join(UPLOAD_DIR, stored_name)
    bytes_written = 0

    try:
        with open(file_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"PDF is too large. Maximum size is {MAX_UPLOAD_MB} MB.",
                    )
                buffer.write(chunk)

        with open(file_path, "rb") as buffer:
            if buffer.read(5) != b"%PDF-":
                raise HTTPException(status_code=400, detail="Uploaded file is not a valid PDF.")
    except Exception:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    finally:
        await file.close()

    return file_path


def _validated_top_k(top_k: int) -> int:
    if top_k < 1 or top_k > MAX_TOP_K:
        raise HTTPException(status_code=400, detail=f"top_k must be between 1 and {MAX_TOP_K}.")
    return top_k


def _validated_query(query: str) -> str:
    normalized = query.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if len(normalized) > MAX_QUERY_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"Query is too long. Maximum length is {MAX_QUERY_CHARS} characters.",
        )
    return normalized


@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")


@app.post("/upload")
async def upload_document(
    request: Request,
    response: Response,
    file: UploadFile = File(...),
):
    """Accepts a PDF upload, saves it safely, and extracts text from it."""
    safe_name = _validate_pdf_name(file.filename)
    file_path = await _save_limited_upload(file, safe_name)

    try:
        pages = extract_text_from_pdf(file_path)
    except Exception as exc:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail=f"Could not read PDF: {exc}") from exc

    if not pages:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=400, detail="Could not extract text. Is the PDF scanned/image-based?")
    if len(pages) > MAX_PDF_PAGES:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(
            status_code=413,
            detail=f"PDF has too many text pages. Maximum is {MAX_PDF_PAGES}.",
        )

    document_store = _get_document_store(request, response)
    previous_path = document_store.get("stored_path")
    if previous_path and os.path.exists(previous_path) and previous_path != file_path:
        os.remove(previous_path)

    document_store.update({
        "filename": safe_name,
        "stored_path": file_path,
        "pages": pages,
        "chunks": [],
        "index": None,
        "last_answer": {},
        "history": [],
        "cache": {},
    })

    return {
        "message": "Document uploaded and text extracted successfully.",
        "filename": safe_name,
        "total_pages": len(pages),
        "preview": pages[0]["text"][:300] + "...",
    }


@app.get("/document/info")
def document_info(request: Request, response: Response):
    """Returns info about the currently loaded document."""
    document_store = _get_document_store(request, response)
    if not document_store["filename"]:
        raise HTTPException(status_code=404, detail="No document uploaded yet.")

    return {
        "filename": document_store["filename"],
        "total_pages": len(document_store["pages"]),
        "total_chunks": len(document_store["chunks"]),
        "index_ready": document_store["index"] is not None,
    }


@app.post("/chunk")
def chunk_document(request: Request, response: Response):
    """Chunks the currently loaded document using adaptive chunking."""
    document_store = _get_document_store(request, response)
    if not document_store["pages"]:
        raise HTTPException(status_code=400, detail="No document loaded. Please upload a PDF first.")

    chunks = adaptive_chunk(document_store["pages"])
    document_store["chunks"] = chunks
    document_store["index"] = None
    document_store["cache"] = {}

    word_counts = [c["word_count"] for c in chunks]
    avg_words = sum(word_counts) // len(word_counts) if word_counts else 0

    return {
        "message": "Document chunked successfully.",
        "total_chunks": len(chunks),
        "avg_chunk_words": avg_words,
        "min_chunk_words": min(word_counts) if word_counts else 0,
        "max_chunk_words": max(word_counts) if word_counts else 0,
        "sample_chunk": chunks[0] if chunks else None,
    }


@app.post("/index")
def index_document(request: Request, response: Response):
    """Builds FAISS + BM25 index from the chunked document."""
    document_store = _get_document_store(request, response)
    if not document_store["chunks"]:
        raise HTTPException(status_code=400, detail="No chunks found. Please run /chunk first.")

    index_data = build_index(document_store["chunks"])
    document_store["index"] = index_data

    return {
        "message": "Hybrid index built successfully.",
        "total_vectors": len(document_store["chunks"]),
        "index_type": "FAISS (dense) + BM25 (sparse) with RRF fusion",
    }


@app.post("/retrieve")
def retrieve_chunks(
    request: Request,
    response: Response,
    query: str = Query(...),
    top_k: int = Query(5),
):
    """Retrieves the most relevant chunks for a given query."""
    document_store = _get_document_store(request, response)
    query = _validated_query(query)
    top_k = _validated_top_k(top_k)

    if document_store["index"] is None:
        raise HTTPException(status_code=400, detail="Index not built yet. Please run /index first.")

    results = retrieve(query, document_store["index"], top_k)

    return {
        "query": query,
        "top_k": top_k,
        "retrieved_chunks": results,
    }


@app.post("/ask")
def ask_question(
    request: Request,
    response: Response,
    query: str = Query(...),
    top_k: int = Query(5),
):
    """Full RAG pipeline with caching, re-ranking and deduplication."""
    document_store = _get_document_store(request, response)
    query = _validated_query(query)
    top_k = _validated_top_k(top_k)

    if document_store["index"] is None:
        raise HTTPException(status_code=400, detail="Index not ready. Please upload, chunk and index first.")

    key = _cache_key(query)
    if key in document_store["cache"]:
        cached = document_store["cache"][key]
        document_store["last_answer"] = cached["result"]
        return {
            "query": query,
            "answer": cached["answer"],
            "citations": cached["citations"],
            "chunks_used": cached["chunks_used"],
            "cached": True,
        }

    retrieved = retrieve(query, document_store["index"], top_k)
    result = generate_answer(query, retrieved)
    if result.get("error"):
        raise HTTPException(status_code=502, detail=result["answer"])

    cache_entry = {
        "answer": result["answer"],
        "citations": result["citations"],
        "chunks_used": len(retrieved),
        "result": result,
    }
    document_store["cache"][key] = cache_entry
    document_store["last_answer"] = result
    document_store["history"].append({
        "query": query,
        "answer": result["answer"],
        "citations": result["citations"],
        "timestamp": datetime.utcnow().isoformat() + "Z",
    })

    return {
        "query": query,
        "answer": result["answer"],
        "citations": result["citations"],
        "chunks_used": len(retrieved),
        "cached": False,
    }


@app.get("/history")
def get_history(request: Request, response: Response):
    """Returns the full conversation history for the session."""
    document_store = _get_document_store(request, response)
    return {
        "total_questions": len(document_store["history"]),
        "history": document_store["history"],
    }


@app.post("/reset")
def reset_session(request: Request, response: Response):
    document_store = _get_document_store(request, response)
    _delete_session_file(document_store)
    document_store.clear()
    document_store.update(_new_document_store())
    return {"message": "Session reset successfully."}


@app.get("/citations")
def get_citations(request: Request, response: Response):
    """Returns the full cited passages from the last /ask call."""
    document_store = _get_document_store(request, response)
    result = document_store.get("last_answer") or {}
    citations = result.get("citations")
    if citations is None:
        raise HTTPException(status_code=404, detail="No answer generated yet. Call /ask first.")

    return {
        "total_citations": len(citations),
        "citations": citations,
    }


@app.get("/stats")
def get_stats(request: Request, response: Response):
    """Returns system performance statistics for the current session."""
    document_store = _get_document_store(request, response)
    chunks = document_store["chunks"]
    history = document_store["history"]

    if not chunks:
        return {"message": "No document loaded yet."}

    word_counts = [c["word_count"] for c in chunks]

    return {
        "document": {
            "filename": document_store["filename"],
            "total_pages": len(document_store["pages"]),
            "total_chunks": len(chunks),
            "avg_chunk_words": round(sum(word_counts) / len(word_counts), 1),
            "min_chunk_words": min(word_counts),
            "max_chunk_words": max(word_counts),
        },
        "session": {
            "questions_asked": len(history),
            "cache_size": len(document_store["cache"]),
        },
        "system": {
            "retrieval_method": "Dense (FAISS) + Sparse (BM25) + RRF Fusion",
            "chunking_method": "Adaptive semantic boundary detection",
            "llm_model": "GLM-4.5-Air via OpenRouter",
            "embedding_model": "all-MiniLM-L6-v2 (384 dims)",
        },
    }


@app.get("/test/health")
def health_check(request: Request, response: Response):
    """Full system health check for the current session."""
    document_store = _get_document_store(request, response)
    results = {
        "document_loaded": {
            "status": "pass" if document_store["filename"] else "fail",
            "detail": document_store["filename"] or "No document uploaded",
        },
        "chunking": {
            "status": "pass" if document_store["chunks"] else "fail",
            "detail": f"{len(document_store['chunks'])} chunks" if document_store["chunks"] else "Not chunked yet",
        },
        "index": {
            "status": "pass" if document_store["index"] else "fail",
            "detail": "FAISS + BM25 ready" if document_store["index"] else "Not indexed yet",
        },
        "cache": {
            "status": "pass",
            "detail": f"{len(document_store['cache'])} queries cached",
        },
        "history": {
            "status": "pass",
            "detail": f"{len(document_store['history'])} questions answered",
        },
    }

    all_pass = all(v["status"] == "pass" for v in results.values())

    return {
        "overall": "healthy" if all_pass else "degraded",
        "checks": results,
    }
