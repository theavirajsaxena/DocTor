# retriever.py
import numpy as np
import faiss
from indexer import get_embedding_model


def _reciprocal_rank_fusion(
    dense_ids : list[int],
    sparse_ids: list[int],
    k         : int = 60
) -> dict:
    scores = {}
    for rank, idx in enumerate(dense_ids):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, idx in enumerate(sparse_ids):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    return scores


def _rerank_by_coverage(
    query : str,
    chunks: list[dict],
    scores: dict
) -> list[int]:
    query_terms = set(query.lower().split())
    boosted = {}
    for chunk_id, rrf_score in scores.items():
        if chunk_id >= len(chunks):
            continue
        chunk_text = chunks[chunk_id]["text"].lower()
        term_hits  = sum(1 for t in query_terms if t in chunk_text)
        coverage   = term_hits / max(len(query_terms), 1)
        boosted[chunk_id] = rrf_score * (1 + 0.1 * coverage)
    return sorted(boosted, key=lambda x: boosted[x], reverse=True)


def _deduplicate_chunks(
    chunk_ids: list[int],
    chunks   : list[dict],
    threshold: float = 0.85
) -> list[int]:
    selected   = []
    seen_words = []
    for cid in chunk_ids:
        if cid >= len(chunks):
            continue
        words = set(chunks[cid]["text"].lower().split())
        is_duplicate = False
        for seen in seen_words:
            if len(words) == 0 or len(seen) == 0:
                continue
            overlap = len(words & seen) / len(words | seen)
            if overlap > threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            selected.append(cid)
            seen_words.append(words)
    return selected


def retrieve(
    query     : str,
    index_data: dict,
    top_k     : int = 5
) -> list[dict]:
    chunks    = index_data["chunks"]
    faiss_idx = index_data["faiss"]
    bm25_idx  = index_data["bm25"]

    # Dense retrieval
    model = get_embedding_model()
    query_vec = model.encode(
        [query], convert_to_numpy=True
    ).astype("float32")
    faiss.normalize_L2(query_vec)

    n_candidates     = min(30, len(chunks))
    _, dense_indices = faiss_idx.search(query_vec, n_candidates)
    dense_ids        = [int(i) for i in dense_indices[0] if i >= 0]

    # Sparse BM25
    bm25_scores = bm25_idx.get_scores(query.lower().split())
    sparse_ids  = np.argsort(bm25_scores)[::-1][:30].tolist()

    # RRF fusion
    rrf_scores   = _reciprocal_rank_fusion(dense_ids, sparse_ids)
    reranked_ids = _rerank_by_coverage(query, chunks, rrf_scores)
    unique_ids   = _deduplicate_chunks(reranked_ids, chunks)

    results = []
    for rank, chunk_id in enumerate(unique_ids[:top_k]):
        chunk = chunks[chunk_id].copy()
        chunk["retrieval_rank"] = rank + 1
        chunk["rrf_score"]      = round(rrf_scores.get(chunk_id, 0), 6)
        results.append(chunk)

    return results
