# Repository Guidelines

## Project Structure & Module Organization
This repository is a small FastAPI-based RAG application with a flat Python layout at the repo root. `main.py` exposes the API and serves `static/index.html`. Core pipeline modules live beside it: `extractor.py` handles PDF text extraction, `chunker.py` builds chunks, `indexer.py` creates FAISS/BM25 indexes, `retriever.py` ranks results, and `generator.py` calls the LLM. Test coverage is currently concentrated in `test_doctor.py`, which exercises the running API end to end. Runtime data belongs in `uploads/`; generated caches, `__pycache__/`, and `venv/` should stay out of commits.

## Build, Test, and Development Commands
Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the API locally with hot reload:

```powershell
uvicorn main:app --reload
```

Open the UI at `http://127.0.0.1:8000/static/index.html` or inspect endpoints at `http://127.0.0.1:8000/docs`. Run the current regression script with the server already running:

```powershell
python test_doctor.py
```

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, snake_case for functions and variables, and short module names matching responsibilities such as `retriever.py`. Keep FastAPI handlers thin and move reusable logic into helper modules. Prefer small docstrings on public functions and return plain dicts with stable keys for API responses. No formatter or linter config is checked in, so match surrounding code closely and keep imports grouped and readable.

## Testing Guidelines
`test_doctor.py` is an integration-style script, not a pytest suite. It expects a live server at `127.0.0.1:8000` and at least one sample PDF in `uploads/`. When adding endpoints or changing response shapes, extend this script with clear labeled checks and keep names aligned with the API behavior being verified.

## Commit & Pull Request Guidelines
Git history is not available in this workspace, so use short imperative commit subjects such as `Add cache stats endpoint` or `Tighten PDF upload validation`. Keep commits focused. PRs should describe the user-visible change, list any new environment variables, include API examples for changed endpoints, and attach UI screenshots when `static/index.html` is modified.

## Security & Configuration Tips
Store secrets in `.env`, currently including `OPENROUTER_API_KEY`, and never hard-code keys. Avoid committing real PDFs from `uploads/` unless they are sanitized test fixtures.
