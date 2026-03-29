# retriever.py
import numpy as np
import faiss
from indexer import embedding_model


def _reciprocal_rank_fusion(
    dense_ids : list[int],
    sparse_ids: list[int],
    k         : int = 60
) -> dict[int, float]:
    """
    Combines two ranked lists using Reciprocal Rank Fusion.
    Returns a dict of {chunk_id: rrf_score}.
    """
    scores = {}
    for rank, idx in enumerate(dense_ids):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    for rank, idx in enumerate(sparse_ids):
        scores[idx] = scores.get(idx, 0) + 1 / (k + rank + 1)
    return scores


def _rerank_by_coverage(
    query  : str,
    chunks : list[dict],
    scores : dict[int, float]
) -> list[int]:
    """
    Applies a lightweight coverage boost on top of RRF scores.
    Chunks that contain more query terms get a small bonus.
    This improves precision for multi-term technical queries.
    """
    query_terms = set(query.lower().split())
    boosted = {}

    for chunk_id, rrf_score in scores.items():
        idx = chunk_id
        if idx >= len(chunks):
            continue
        chunk_text  = chunks[idx]["text"].lower()
        # Count how many unique query terms appear in this chunk
        term_hits   = sum(1 for t in query_terms if t in chunk_text)
        coverage    = term_hits / max(len(query_terms), 1)
        # Small 10% boost based on term coverage
        boosted[idx] = rrf_score * (1 + 0.1 * coverage)

    return sorted(boosted, key=lambda x: boosted[x], reverse=True)


def _deduplicate_chunks(
    chunk_ids : list[int],
    chunks    : list[dict],
    threshold : float = 0.85
) -> list[int]:
    """
    Removes near-duplicate chunks from the result set.
    If two chunks share more than `threshold` of their words,
    only the higher-ranked one is kept.
    This prevents the LLM receiving repetitive context.
    """
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
    """
    Hybrid retrieval pipeline:
    1. Dense FAISS search (semantic)
    2. Sparse BM25 search (keyword)
    3. Reciprocal Rank Fusion
    4. Coverage-based re-ranking boost
    5. Deduplication
    6. Return top_k results
    """
    chunks    = index_data["chunks"]
    faiss_idx = index_data["faiss"]
    bm25_idx  = index_data["bm25"]

    # ── 1. Dense retrieval ─────────────────────────────────────────
    query_vec = embedding_model.encode(
        [query], convert_to_numpy=True
    ).astype("float32")
    faiss.normalize_L2(query_vec)

    n_candidates = min(30, len(chunks))  # cast wider net
    _, dense_indices = faiss_idx.search(query_vec, n_candidates)
    dense_ids = [int(i) for i in dense_indices[0] if i >= 0]

    # ── 2. Sparse BM25 retrieval ───────────────────────────────────
    bm25_scores = bm25_idx.get_scores(query.lower().split())
    sparse_ids  = np.argsort(bm25_scores)[::-1][:30].tolist()

    # ── 3. RRF fusion ──────────────────────────────────────────────
    rrf_scores  = _reciprocal_rank_fusion(dense_ids, sparse_ids)

    # ── 4. Coverage re-ranking ─────────────────────────────────────
    reranked_ids = _rerank_by_coverage(query, chunks, rrf_scores)

    # ── 5. Deduplicate ─────────────────────────────────────────────
    unique_ids = _deduplicate_chunks(reranked_ids, chunks)

    # ── 6. Build result objects ────────────────────────────────────
    results = []
    for rank, chunk_id in enumerate(unique_ids[:top_k]):
        chunk = chunks[chunk_id].copy()
        chunk["retrieval_rank"] = rank + 1
        chunk["rrf_score"]      = round(rrf_scores.get(chunk_id, 0), 6)
        results.append(chunk)

    return results