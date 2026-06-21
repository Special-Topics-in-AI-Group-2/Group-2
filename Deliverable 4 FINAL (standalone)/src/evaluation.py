"""Retriever evaluation harness for CSAI415 D1.

evaluate_retriever() is the single entry-point expected by make_optuna_objective:

    metrics = evaluate_retriever(retriever, queries, k=5)

It returns a dict with exactly the keys the Optuna objective and run card
builders look for:

    {
        "ndcg@5":        float,   # mean NDCG@k across all queries
        "recall@5":      float,   # mean Recall@k across all queries
        "mrr":           float,   # Mean Reciprocal Rank@k
        "p95_ms":        float,   # p95 latency across all timed search calls
        "mean_ms":       float,   # mean latency across all timed search calls
    }

Latency measurement design
--------------------------
Each query is run `repeats` times (default 5).  This gives 40 * 5 = 200 timed
observations over the gold set, placing p95 at the 190th value — a real
percentile rather than the 2nd-highest out of 40.  See the discussion in
docs/d1_notes.md for why n=40 is too small for a stable p95.

Only retriever.search() is timed.  Corpus fitting (retriever.fit()) is a
one-time offline cost and is excluded.

A single warmup pass over all queries is performed before timing begins.
This absorbs:
  - sklearn's internal caching on the first kneighbors() call
  - TF-IDF vocabulary lookup overhead on the first transform() call
  - Python's function dispatch warm-up

time.perf_counter() is used throughout — it is monotonic, high-resolution,
and unaffected by NTP / system clock adjustments.

BM25 integration
----------------
Pass bm25_fn to enable hybrid evaluation.  The function receives a query
string and must return a dict[chunk_id -> raw_bm25_score].  When None
(default), the retriever runs in dense-only mode and WSF/RRF fusion is not
applied.  This matches the D1 baseline where BM25 is not yet integrated.

Quality vs latency
------------------
Quality metrics (NDCG, Recall, MRR) are computed from the *first* repeat
only.  Subsequent repeats exist purely for timing stability — running the
same query r times and averaging the retrieved lists would not change quality
scores because the retriever is deterministic.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Union

import numpy as np

from src.data_utils import Query
from src.metrics import evaluate_ranking
from src.retriever import HybridKNNRetriever


# ---------------------------------------------------------------------------
# type aliases
# ---------------------------------------------------------------------------

BM25Fn = Callable[[str], Dict[str, float]]
MetricsDict = Dict[str, float]


# ---------------------------------------------------------------------------
# public entry-point
# ---------------------------------------------------------------------------

def evaluate_retriever(
    retriever: HybridKNNRetriever,
    queries: List[Query],
    *,
    k: int = 5,
    repeats: int = 5,
    bm25_fn: Optional[BM25Fn] = None,
    rrf_k: Optional[int] = None,
) -> MetricsDict:
    """Evaluate a fitted HybridKNNRetriever on a gold query set.

    Parameters
    ----------
    retriever : HybridKNNRetriever
        A retriever that has already been fitted with retriever.fit(chunks).
        This function never calls fit() — corpus setup is the caller's
        responsibility so it is not included in latency measurements.
    queries : list[Query]
        Gold query set.  Each Query carries query_text and relevant_chunk_ids.
        Typically the 40-query set from build_corpus() or a held-out subset.
    k : int
        Rank cut-off for all metrics.  Default 5 (NDCG@5, Recall@5, MRR@5).
    repeats : int
        Number of timed search passes per query.  Default 5.
        Total timed observations = len(queries) * repeats.
        Minimum recommended: 3.  Use 10 for publication-grade p95 estimates.
    bm25_fn : callable | None
        Optional function (query_text) -> dict[chunk_id, float] providing
        raw BM25 scores for hybrid fusion.  When None, dense-only mode is
        used and bm25_score_norm / fused_score fields of HybridResult will
        be None.
    rrf_k : int | None
        When provided alongside bm25_fn, Reciprocal Rank Fusion is used
        instead of WSF.  Standard value is 60 (Cormack 2009).
        Ignored when bm25_fn is None.

    Returns
    -------
    dict with keys:
        "ndcg@{k}"  : mean NDCG@k across all queries          (float, [0,1])
        "recall@{k}": mean Recall@k across all queries         (float, [0,1])
        "mrr"       : Mean Reciprocal Rank@k                   (float, [0,1])
        "p95_ms"    : 95th-percentile search latency in ms     (float, ≥ 0)
        "mean_ms"   : mean search latency in ms                (float, ≥ 0)

    Raises
    ------
    ValueError
        If queries is empty or repeats < 1.
    RuntimeError
        If retriever has not been fitted (propagated from retriever.search).

    Notes
    -----
    The dict keys "ndcg@{k}" and "recall@{k}" use the actual k value, so
    evaluate_retriever(..., k=5) returns "ndcg@5" and "recall@5" — matching
    the keys expected by make_optuna_objective and build_run_card.
    """
    if not queries:
        raise ValueError("queries must be non-empty.")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}.")

    # ── 0. Warmup — never timed ───────────────────────────────────────────────
    # One pass absorbs sklearn's internal caching and TF-IDF vocabulary
    # lookup overhead that does not reflect steady-state search cost.
    _warmup(retriever, queries, bm25_fn=bm25_fn, rrf_k=rrf_k, k=k)

    # ── 1. First repeat: collect both quality results and timings ─────────────
    retrieved_lists: List[List[str]] = []
    relevant_sets:   List[set]       = []
    timings_ms:      List[float]     = []

    for query in queries:
        bm25_scores = bm25_fn(query.query_text) if bm25_fn is not None else None

        t0 = time.perf_counter()
        results = retriever.search(
            query.query_text,
            top_k=k,
            bm25_scores=bm25_scores,
            rrf_k=rrf_k,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1_000

        timings_ms.append(elapsed_ms)
        retrieved_lists.append([r.chunk.chunk_id for r in results])
        relevant_sets.append(set(query.relevant_chunk_ids))

    # ── 2. Remaining repeats: timing only ─────────────────────────────────────
    # Quality metrics are deterministic for a fixed retriever — re-running
    # would produce identical retrieved lists.  Only latency needs more samples.
    for _ in range(repeats - 1):
        for query in queries:
            bm25_scores = bm25_fn(query.query_text) if bm25_fn is not None else None

            t0 = time.perf_counter()
            retriever.search(
                query.query_text,
                top_k=k,
                bm25_scores=bm25_scores,
                rrf_k=rrf_k,
            )
            timings_ms.append((time.perf_counter() - t0) * 1_000)

    # ── 3. Quality metrics ────────────────────────────────────────────────────
    quality = evaluate_ranking(retrieved_lists, relevant_sets, k=k)

    # ── 4. Latency statistics ─────────────────────────────────────────────────
    timings = np.array(timings_ms, dtype=float)
    p95_ms  = float(np.percentile(timings, 95))
    mean_ms = float(np.mean(timings))

    return {
        **quality,                  # ndcg@k, recall@k, mrr
        "p95_ms":  p95_ms,
        "mean_ms": mean_ms,
    }


# ---------------------------------------------------------------------------
# private helpers
# ---------------------------------------------------------------------------

def _warmup(
    retriever: HybridKNNRetriever,
    queries: List[Query],
    *,
    bm25_fn: Optional[BM25Fn],
    rrf_k: Optional[int],
    k: int,
) -> None:
    """Run one untimed pass over all queries to prime internal caches."""
    for query in queries:
        bm25_scores = bm25_fn(query.query_text) if bm25_fn is not None else None
        retriever.search(
            query.query_text,
            top_k=k,
            bm25_scores=bm25_scores,
            rrf_k=rrf_k,
        )


# ---------------------------------------------------------------------------
# per-query breakdown (optional — useful for D1 report analysis)
# ---------------------------------------------------------------------------

def evaluate_retriever_per_query(
    retriever: HybridKNNRetriever,
    queries: List[Query],
    *,
    k: int = 5,
    repeats: int = 5,
    bm25_fn: Optional[BM25Fn] = None,
    rrf_k: Optional[int] = None,
) -> List[Dict[str, Union[str, float, List[str]]]]:
    """Per-query breakdown of retrieval quality and latency.

    Returns one dict per query — useful for inspecting failure cases and
    building the top-k examples table required by D2.

    Each dict contains:
        query_id        : str
        topic_id        : str
        query_text      : str
        query_type      : str
        retrieved_ids   : list[str]   top-k chunk IDs
        relevant_ids    : list[str]   ground-truth IDs
        ndcg            : float       NDCG@k for this query
        recall          : float       Recall@k for this query
        reciprocal_rank : float       RR for this query
        mean_ms         : float       mean latency over repeats (ms)

    Parameters match evaluate_retriever().
    """
    if not queries:
        raise ValueError("queries must be non-empty.")
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}.")

    from src.metrics import ndcg_at_k, recall_at_k, reciprocal_rank as rr_fn

    _warmup(retriever, queries, bm25_fn=bm25_fn, rrf_k=rrf_k, k=k)

    rows = []

    for query in queries:
        relevant = set(query.relevant_chunk_ids)
        bm25_scores = bm25_fn(query.query_text) if bm25_fn is not None else None

        # Collect `repeats` timings; use retrieved list from the first run.
        retrieved_ids: List[str] = []
        query_timings: List[float] = []

        for rep in range(repeats):
            t0 = time.perf_counter()
            results = retriever.search(
                query.query_text,
                top_k=k,
                bm25_scores=bm25_scores,
                rrf_k=rrf_k,
            )
            query_timings.append((time.perf_counter() - t0) * 1_000)

            if rep == 0:
                retrieved_ids = [r.chunk.chunk_id for r in results]

        rows.append({
            "query_id":        query.query_id,
            "topic_id":        query.topic_id,
            "query_text":      query.query_text,
            "query_type":      query.query_type,
            "retrieved_ids":   retrieved_ids,
            "relevant_ids":    list(relevant),
            "ndcg":            ndcg_at_k(retrieved_ids, relevant, k),
            "recall":          recall_at_k(retrieved_ids, relevant, k),
            "reciprocal_rank": rr_fn(retrieved_ids, relevant, k),
            "mean_ms":         float(np.mean(query_timings)),
        })

    return rows
