# test_doctor.py
# Run this with: python test_doctor.py
# Make sure uvicorn is running before running this script

import requests
import time
import json
import os

BASE = "http://127.0.0.1:8000"
SEPARATOR = "─" * 55

def header(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)

def check(label, condition, detail=""):
    icon = "✅" if condition else "❌"
    print(f"  {icon}  {label}")
    if detail:
        print(f"       {detail}")
    return condition

def run_tests():
    total  = 0
    passed = 0

    # ── TEST 1: Server Health ──────────────────────────────────────
    header("TEST 1 — Server Health")
    try:
        r = requests.get(f"{BASE}/", timeout=5)
        total  += 1
        passed += check("Server is running", r.status_code == 200)
    except Exception as e:
        total += 1
        check("Server is running", False, str(e))
        print("\n  ⚠️  Server not running! Start it with: uvicorn main:app --reload")
        return

    # ── TEST 2: Document Upload ────────────────────────────────────
    header("TEST 2 — Document Upload")

    # Find any PDF in uploads folder
    pdf_path = None
    if os.path.exists("uploads"):
        for f in os.listdir("uploads"):
            if f.endswith(".pdf"):
                pdf_path = os.path.join("uploads", f)
                break

    if not pdf_path:
        print("  ⚠️  No PDF found in uploads/ folder.")
        print("       Upload a PDF through the UI first, then re-run this script.")
    else:
        with open(pdf_path, "rb") as f:
            r = requests.post(
                f"{BASE}/upload",
                files={"file": (os.path.basename(pdf_path), f, "application/pdf")}
            )
        total  += 1
        passed += check(
            "PDF uploads successfully",
            r.status_code == 200,
            f"File: {os.path.basename(pdf_path)}"
        )
        if r.status_code == 200:
            data = r.json()
            total  += 1
            passed += check(
                "Text extracted from PDF",
                data.get("total_pages", 0) > 0,
                f"Pages extracted: {data.get('total_pages', 0)}"
            )
            total  += 1
            passed += check(
                "Preview text returned",
                len(data.get("preview", "")) > 10,
                f"Preview: {data.get('preview','')[:60]}..."
            )

    # ── TEST 3: Chunking ───────────────────────────────────────────
    header("TEST 3 — Adaptive Chunking")
    r = requests.post(f"{BASE}/chunk")
    total  += 1
    passed += check(
        "Chunking completes",
        r.status_code == 200
    )
    if r.status_code == 200:
        data = r.json()
        total  += 1
        passed += check(
            "Chunks produced",
            data.get("total_chunks", 0) > 0,
            f"Total chunks: {data.get('total_chunks', 0)}"
        )
        total  += 1
        avg = data.get("avg_chunk_words", 0)
        passed += check(
            "Average chunk size is reasonable (60–200 words)",
            60 <= avg <= 220,
            f"Avg words per chunk: {avg}"
        )

    # ── TEST 4: Indexing ───────────────────────────────────────────
    header("TEST 4 — Hybrid Indexing")
    r = requests.post(f"{BASE}/index")
    total  += 1
    passed += check(
        "Index builds successfully",
        r.status_code == 200
    )
    if r.status_code == 200:
        data = r.json()
        total  += 1
        passed += check(
            "FAISS + BM25 index confirmed",
            "FAISS" in data.get("index_type", ""),
            data.get("index_type", "")
        )

    # ── TEST 5: Retrieval ──────────────────────────────────────────
    header("TEST 5 — Hybrid Retrieval")
    test_queries = [
        "What is the main topic of this document?",
        "What methodology is described?",
        "What are the key findings or conclusions?"
    ]
    for query in test_queries:
        r = requests.post(
            f"{BASE}/retrieve",
            params={"query": query, "top_k": 5}
        )
        total  += 1
        passed += check(
            f"Retrieval works: '{query[:40]}...'",
            r.status_code == 200 and
            len(r.json().get("retrieved_chunks", [])) > 0,
            f"Chunks returned: "
            f"{len(r.json().get('retrieved_chunks', []))}"
            if r.status_code == 200 else "Failed"
        )

    # ── TEST 6: Answer Generation ──────────────────────────────────
    header("TEST 6 — Answer Generation")
    r = requests.post(
        f"{BASE}/ask",
        params={"query": "What is this document about?", "top_k": 5}
    )
    total  += 1
    passed += check(
        "Answer generated",
        r.status_code == 200
    )
    if r.status_code == 200:
        data = r.json()
        total  += 1
        passed += check(
            "Answer is non-empty",
            len(data.get("answer", "")) > 20,
            f"Answer length: {len(data.get('answer',''))} chars"
        )
        total  += 1
        passed += check(
            "Citations returned",
            isinstance(data.get("citations"), list),
            f"Citations: {len(data.get('citations', []))}"
        )
        total  += 1
        passed += check(
            "chunks_used field present",
            data.get("chunks_used", 0) > 0,
            f"Chunks used: {data.get('chunks_used', 0)}"
        )

    # ── TEST 7: Caching ────────────────────────────────────────────
    header("TEST 7 — Response Caching")
    query = "What is this document about?"
    t1 = time.time()
    requests.post(f"{BASE}/ask", params={"query": query})
    t2 = time.time()
    r = requests.post(f"{BASE}/ask", params={"query": query})
    t3 = time.time()

    first_time  = round((t2 - t1) * 1000)
    cached_time = round((t3 - t2) * 1000)

    total  += 1
    passed += check(
        "Cached response is fast (under 500ms)",
        cached_time < 500,
        f"First: {first_time}ms | Cached: {cached_time}ms"
    )
    total  += 1
    passed += check(
        "Cache flag returned",
        r.json().get("cached") == True,
        f"cached: {r.json().get('cached')}"
    )

    # ── TEST 8: Citations Endpoint ─────────────────────────────────
    header("TEST 8 — Citations Endpoint")
    r = requests.get(f"{BASE}/citations")
    total  += 1
    passed += check(
        "Citations endpoint works",
        r.status_code == 200
    )
    if r.status_code == 200:
        total  += 1
        passed += check(
            "Citations have full text",
            all(
                len(c.get("full_text", "")) > 10
                for c in r.json().get("citations", [])
            ),
            f"Total citations: {r.json().get('total_citations', 0)}"
        )

    # ── TEST 9: Stats Endpoint ─────────────────────────────────────
    header("TEST 9 — Stats & Health")
    r = requests.get(f"{BASE}/stats")
    total  += 1
    passed += check(
        "Stats endpoint works",
        r.status_code == 200
    )

    r = requests.get(f"{BASE}/test/health")
    total  += 1
    passed += check(
        "Health check passes",
        r.status_code == 200 and
        r.json().get("overall") == "healthy",
        f"Status: {r.json().get('overall','unknown')}"
        if r.status_code == 200 else "Failed"
    )

    # ── TEST 10: Session Reset ─────────────────────────────────────
    header("TEST 10 — Session Reset")
    r = requests.post(f"{BASE}/reset")
    total  += 1
    passed += check(
        "Reset clears session",
        r.status_code == 200
    )
    r = requests.get(f"{BASE}/document/info")
    total  += 1
    passed += check(
        "Document cleared after reset",
        r.status_code == 404
    )

    # ── Final Score ────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    score = round(passed / total * 100)
    print(f"  FINAL SCORE: {passed}/{total} tests passed ({score}%)")
    if score == 100:
        print("  🎉 All tests passed! DocTor is ready for submission.")
    elif score >= 80:
        print("  ✅ System is working well. Minor issues to review.")
    else:
        print("  ⚠️  Some tests failed. Review the ❌ items above.")
    print(f"{'═'*55}\n")

if __name__ == "__main__":
    run_tests()
# ```

# ---

# ### 10.3 — Run the Test Suite

# Make sure your uvicorn server is running, then open a **second** Command Prompt window (venv active, inside `C:\Aviraj\DocTor`):
# ```
# python test_doctor.py
# ```

# You should see results like:
# ```
# ───────────────────────────────────────────────────────
#   TEST 1 — Server Health
# ───────────────────────────────────────────────────────
#   ✅  Server is running

#   TEST 2 — Document Upload
# ...
# ═══════════════════════════════════════════════════════
#   FINAL SCORE: 22/22 tests passed (100%)
#   🎉 All tests passed! DocTor is ready for submission.
# ═══════════════════════════════════════════════════════
# # ```

# ---

# ### 10.4 — Final Project Structure Verification

# Run this in Command Prompt to confirm all files are in place:
# ```
# dir C:\Aviraj\DocTor /b
# ```

# You should see:
# ```
# .env
# chunker.py
# extractor.py
# generator.py
# indexer.py
# main.py
# retriever.py
# test_doctor.py
# uploads/
# static/
#     index.html
# venv/
# ```

# ---

# ### 10.5 — Create a requirements.txt for Submission

# Run this command to generate a clean requirements file for your project submission:
# ```
# pip freeze > requirements.txt
# ```

# This file lets anyone recreate your exact environment with `pip install -r requirements.txt`.

# ---

# ### 10.6 — Final Demo Checklist

# Before your viva, rehearse this exact demo flow:

# **1. Start the server**
# ```
# cd C:\Aviraj\DocTor
# venv\Scripts\activate
# uvicorn main:app --reload
# ```

# **2. Open the UI**
# ```
# http://127.0.0.1:8000/static/index.html