"""Metadata-filtered hybrid (BM25 + TF-IDF cosine) retrieval via reciprocal rank fusion."""

from __future__ import annotations

from .index import BM25, reciprocal_rank_fusion, tfidf_embed, vector_search

# How many candidates each ranker contributes to fusion before truncating to
# top_k. Needs to be wide enough that a strong BM25 match isn't zeroed out by
# RRF just because TF-IDF cosine (weak on generic-word-heavy questions against
# boilerplate-heavy insurance docs) ranks it outside a too-tight cutoff --
# found via Phase 1 Step 1.7 verification, see outputs/phase1_eval.jsonl.
RETRIEVAL_POOL = 30


def retrieve(
    query: str,
    chunks: list[dict],
    indexes: dict,
    top_k: int = 5,
    insurer: str | None = None,
    product: str | None = None,
    doc_type: str | None = None,
) -> list[dict]:
    if insurer is None and product is None and doc_type is None:
        bm25_results = indexes["bm25"].search(query, top_k=RETRIEVAL_POOL)
        query_emb = tfidf_embed(query, indexes["vocab"], indexes["idf"])
        vec_results = vector_search(query_emb, indexes["embeddings"], top_k=RETRIEVAL_POOL)
        fused = reciprocal_rank_fusion([vec_results, bm25_results])[:top_k]
        return [chunks[i] for i, _score in fused]

    # metadata filter first -- narrows the pool before ranking
    keep_idx = [
        i for i, c in enumerate(chunks)
        if (insurer is None or c["insurer"].lower() == insurer.lower())
        and (product is None or c["product"].lower() == product.lower())
        and (doc_type is None or c["doc_type"].lower() == doc_type.lower())
    ]
    if not keep_idx:
        return []

    filtered_texts = [chunks[i]["text"] for i in keep_idx]
    filtered_embeddings = [indexes["embeddings"][i] for i in keep_idx]

    # BM25's IDF is corpus-dependent, so re-indexing on just the filtered
    # subset gives more accurate term weighting than filtering after ranking
    # against the full corpus. Costs a small re-index per query -- fine at
    # this scale (a few hundred chunks per insurer/product).
    bm25_local = BM25()
    bm25_local.index(filtered_texts)
    bm25_results = bm25_local.search(query, top_k=RETRIEVAL_POOL)  # local indices

    query_emb = tfidf_embed(query, indexes["vocab"], indexes["idf"])
    vec_results = vector_search(query_emb, filtered_embeddings, top_k=RETRIEVAL_POOL)  # local indices

    fused = reciprocal_rank_fusion([vec_results, bm25_results])[:top_k]
    return [chunks[keep_idx[local_i]] for local_i, _score in fused]
