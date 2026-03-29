# indexer.py
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

# Load the embedding model once at startup (not on every request)
# This model converts text into 384-dimensional vectors
print("Loading embedding model... (first time may take a minute)")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("Embedding model loaded.")


def _tokenize(text: str) -> list[str]:
    """Simple whitespace tokenizer for BM25."""
    return text.lower().split()


def build_index(chunks: list[dict]) -> dict:
    """
    Builds both a FAISS dense index and a BM25 sparse index
    from the list of chunks.
    Returns a dict holding both indexes and the original chunks.
    """
    texts = [chunk["text"] for chunk in chunks]

    # ── BM25 sparse index ─────────────────────────────────────────
    tokenized = [_tokenize(t) for t in texts]
    bm25_index = BM25Okapi(tokenized)

    # ── Dense embeddings + FAISS index ───────────────────────────
    print(f"Encoding {len(texts)} chunks into embeddings...")
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    embeddings = embeddings.astype("float32")

    # Normalize vectors so cosine similarity = dot product
    faiss.normalize_L2(embeddings)

    # Build a flat (exact search) FAISS index
    dimension = embeddings.shape[1]           # 384 for MiniLM
    faiss_index = faiss.IndexFlatIP(dimension) # IP = Inner Product
    faiss_index.add(embeddings)

    print(f"Index built: {faiss_index.ntotal} vectors in FAISS.")

    return {
        "faiss"      : faiss_index,
        "bm25"       : bm25_index,
        "embeddings" : embeddings,
        "chunks"     : chunks        # keep reference for retrieval
    }