import os

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", os.path.join(os.getcwd(), "model_cache"))
os.environ.setdefault("HF_HOME", MODEL_CACHE_DIR)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", MODEL_CACHE_DIR)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
embedding_model = None


def get_embedding_model() -> SentenceTransformer:
    global embedding_model
    if embedding_model is None:
        print("Loading embedding model... (first time may take a minute)")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("Embedding model loaded.")
    return embedding_model


def _tokenize(text: str) -> list[str]:
    return text.lower().split()


def build_index(chunks: list[dict]) -> dict:
    texts = [chunk["text"] for chunk in chunks]
    model = get_embedding_model()

    # BM25
    tokenized  = [_tokenize(t) for t in texts]
    bm25_index = BM25Okapi(tokenized)

    # Dense embeddings
    print(f"Encoding {len(texts)} chunks into embeddings...")
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    # FAISS index
    dimension   = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dimension)
    faiss_index.add(embeddings)

    print(f"Index built: {faiss_index.ntotal} vectors in FAISS.")

    return {
        "faiss"      : faiss_index,
        "bm25"       : bm25_index,
        "embeddings" : embeddings,
        "chunks"     : chunks
    }
