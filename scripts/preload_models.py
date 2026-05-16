import os

MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", os.path.join(os.getcwd(), "model_cache"))
os.environ.setdefault("HF_HOME", MODEL_CACHE_DIR)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", MODEL_CACHE_DIR)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import spacy
from sentence_transformers import SentenceTransformer


def main() -> None:
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    spacy.load("en_core_web_sm")
    SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Models are ready in {MODEL_CACHE_DIR}")


if __name__ == "__main__":
    main()
