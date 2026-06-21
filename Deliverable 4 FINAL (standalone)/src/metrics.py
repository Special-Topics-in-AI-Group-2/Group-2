"""Retrieval evaluation metrics for CSAI415 D1.

Three metrics are implemented:

    ndcg_at_k   -- Normalised Discounted Cumulative Gain
    recall_at_k -- Fraction of relevant documents retrieved in top-k
    mean_reciprocal_rank -- Mean of 1/rank_of_first_relevant_doc

All functions follow the same call signature:

    metric(retrieved_ids, relevant_ids, k) -> float

where:
    retrieved_ids : list[str]
        Ranked list of chunk IDs returned by the retriever, most relevant
        first.  Only the first k entries are considered.
    relevant_ids  : set[str] | list[str]
        Ground-truth relevant chunk IDs for this query.
    k             : int
        Cut-off depth.

DCG formula (Järvelin & Kekäläinen 2002, binary relevance variant)
-------------------------------------------------------------------
Rank positions are 1-based.  rel_i ∈ {0, 1}.

    DCG@k = Σ_{i=1}^{k}  rel_i / log2(i + 1)

    i=1 (rank 1) : denominator = log2(2) = 1.0  → full credit, no discount
    i=2 (rank 2) : denominator = log2(3) ≈ 1.58 → ~63 % credit
    i=3 (rank 3) : denominator = log2(4) = 2.0  → 50 % credit
    i=5 (rank 5) : denominator = log2(6) ≈ 2.58 → ~39 % credit

The log2(i+2) form you may have seen is the same formula re-indexed for a
0-based loop variable (i starts at 0, so +2 gives the same denominators).
Both produce identical numbers — the difference is purely notational.

    NDCG@k = DCG@k / IDCG@k

where IDCG@k is the DCG of a perfect ranking: the min(|relevant|, k)
relevant docs placed at positions 1, 2, ..., min(|relevant|, k).

Edge cases
----------
* Empty retrieved list         → 0.0 for all metrics.
* No relevant docs for query   → 0.0 for all metrics (IDCG = 0 → NDCG = 0).
* k > len(retrieved_ids)       → pad with zeros; does not raise.
* MRR: no relevant doc in top-k → contributes 0.0 to the mean.
"""

from __future__ import annotations

import math
from typing import List, Set, Union


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _to_set(ids: Union[List[str], Set[str]]) -> Set[str]:
    return ids if isinstance(ids, set) else set(ids)


def _dcg(ranked_ids: List[str], relevant: Set[str], k: int) -> float:
    """Raw DCG@k using 1-based rank positions and binary relevance.

    DCG@k = Σ_{i=1}^{k}  rel_i / log2(i + 1)

    The first rank (i=1) has denominator log2(2) = 1, so a relevant doc at
    the top contributes exactly 1.0 — no discount.  Every subsequent rank
    applies an increasing logarithmic penalty.
    """
    gain = 0.0
    for i, chunk_id in enumerate(ranked_ids[:k], start=1):   # i is 1-based rank
        if chunk_id in relevant:
            gain += 1.0 / math.log2(i + 1)
    return gain


def _idcg(n_relevant: int, k: int) -> float:
    """Ideal DCG@k: n_relevant docs placed at the top positions.

    IDCG@k = Σ_{i=1}^{min(n_relevant, k)}  1 / log2(i + 1)

    This is the maximum possible DCG@k given the number of relevant docs.
    Dividing DCG@k by IDCG@k normalises the score to [0, 1].
    """
    ideal_positions = min(n_relevant, k)
    return sum(1.0 / math.log2(i + 1) for i in range(1, ideal_positions + 1))


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------

def ndcg_at_k(
    retrieved_ids: List[str],
    relevant_ids: Union[List[str], Set[str]],
    k: int,
) -> float:
    """Normalised Discounted Cumulative Gain at rank k (binary relevance).

    Measures both coverage (did we find the relevant docs?) and ranking
    quality (are they near the top?).  A relevant doc at rank 1 contributes
    the maximum possible gain; later ranks contribute progressively less.

    Parameters
    ----------
    retrieved_ids : list[str]
        Ranked retriever output, most relevant first.
    relevant_ids  : set[str] | list[str]
        Ground-truth relevant chunk IDs.
    k             : int
        Rank cut-off.  Only the first k entries of retrieved_ids are scored.

    Returns
    -------
    float
        NDCG@k in [0.0, 1.0].  Returns 0.0 if relevant_ids is empty
        (undefined case) or if retrieved_ids is empty.

    Examples
    --------
    Perfect ranking — all 3 relevant docs in top 3 positions:
    >>> ndcg_at_k(["a", "b", "c", "d", "e"], {"a", "b", "c"}, k=5)
    1.0

    Relevant docs pushed to the bottom of the window:
    >>> ndcg_at_k(["x", "y", "a", "b", "c"], {"a", "b", "c"}, k=5)
    0.679...   # (exact value depends on k and position)

    No relevant docs retrieved:
    >>> ndcg_at_k(["x", "y", "z"], {"a", "b", "c"}, k=5)
    0.0
    """
    relevant = _to_set(relevant_ids)

    if not relevant or not retrieved_ids:
        return 0.0

    dcg  = _dcg(retrieved_ids, relevant, k)
    idcg = _idcg(len(relevant), k)

    if idcg == 0.0:
        return 0.0

    return dcg / idcg


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

def recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: Union[List[str], Set[str]],
    k: int,
) -> float:
    """Recall at rank k: fraction of relevant docs found in the top-k results.

    Recall@k = |retrieved[:k] ∩ relevant| / |relevant|

    Unlike NDCG, this metric is rank-position blind — it only cares whether
    each relevant doc appears somewhere in the top-k window, not where.

    In our corpus (3 relevant docs per query, k=5), the maximum meaningful
    Recall@5 is 1.0 (all 3 found), and even a weak retriever will tend to
    score well because the window is larger than the relevant set.  Use
    alongside NDCG@5 for a complete picture.

    Parameters
    ----------
    retrieved_ids : list[str]
        Ranked retriever output, most relevant first.
    relevant_ids  : set[str] | list[str]
        Ground-truth relevant chunk IDs.
    k             : int
        Rank cut-off.

    Returns
    -------
    float
        Recall@k in [0.0, 1.0].  Returns 0.0 if relevant_ids is empty.

    Examples
    --------
    All 3 relevant docs found (order doesn't matter for recall):
    >>> recall_at_k(["x", "a", "b", "y", "c"], {"a", "b", "c"}, k=5)
    1.0

    Only 2 of 3 found:
    >>> recall_at_k(["a", "b", "x", "y", "z"], {"a", "b", "c"}, k=5)
    0.6667...
    """
    relevant = _to_set(relevant_ids)

    if not relevant or not retrieved_ids:
        return 0.0

    retrieved_at_k = set(retrieved_ids[:k])
    hits = len(retrieved_at_k & relevant)

    return hits / len(relevant)


# ---------------------------------------------------------------------------
# mean_reciprocal_rank
# ---------------------------------------------------------------------------

def reciprocal_rank(
    retrieved_ids: List[str],
    relevant_ids: Union[List[str], Set[str]],
    k: int,
) -> float:
    """Reciprocal rank of the first relevant document in the top-k results.

    RR = 1 / rank_of_first_relevant_doc,  if found within top-k
       = 0.0,                              otherwise

    Rank positions are 1-based, so a hit at position 1 gives RR = 1.0,
    position 2 gives 0.5, position 5 gives 0.2, etc.

    This is the per-query building block for MRR.  It is sensitive only to
    the *first* relevant hit — useful when users care about whether the
    very first result is relevant (e.g. a direct-answer use case), less so
    when coverage of all relevant docs matters.

    Parameters
    ----------
    retrieved_ids : list[str]
        Ranked retriever output, most relevant first.
    relevant_ids  : set[str] | list[str]
        Ground-truth relevant chunk IDs.
    k             : int
        Only the first k results are considered.

    Returns
    -------
    float
        Reciprocal rank in (0.0, 1.0], or 0.0 if no relevant doc in top-k.
    """
    relevant = _to_set(relevant_ids)

    for i, chunk_id in enumerate(retrieved_ids[:k], start=1):   # i is 1-based
        if chunk_id in relevant:
            return 1.0 / i

    return 0.0


def mean_reciprocal_rank(
    results: List[tuple[List[str], Union[List[str], Set[str]]]],
    k: int,
) -> float:
    """Mean Reciprocal Rank over a list of (retrieved, relevant) query pairs.

    MRR = (1/|Q|) * Σ_q  RR_q

    where RR_q is the reciprocal rank of the first relevant doc for query q.

    MRR is a useful complement to NDCG@5 in our setting: if NDCG improves
    but MRR stays flat, the model found more relevant docs but kept burying
    the best one.  If MRR improves alongside NDCG, ranking quality is
    genuinely better top-to-bottom.

    Parameters
    ----------
    results : list of (retrieved_ids, relevant_ids) tuples
        One tuple per query.
    k       : int
        Rank cut-off applied to every query.

    Returns
    -------
    float
        MRR in [0.0, 1.0].  Returns 0.0 on an empty query list.

    Examples
    --------
    >>> pairs = [
    ...     (["a", "b", "c"], {"a"}),   # first hit at rank 1 → RR = 1.0
    ...     (["x", "a", "c"], {"a"}),   # first hit at rank 2 → RR = 0.5
    ...     (["x", "y", "z"], {"a"}),   # no hit             → RR = 0.0
    ... ]
    >>> mean_reciprocal_rank(pairs, k=5)
    0.5   # (1.0 + 0.5 + 0.0) / 3
    """
    if not results:
        return 0.0

    total_rr = sum(
        reciprocal_rank(retrieved, relevant, k)
        for retrieved, relevant in results
    )

    return total_rr / len(results)


# ---------------------------------------------------------------------------
# convenience: evaluate a full query set in one call
# ---------------------------------------------------------------------------

def evaluate_ranking(
    retrieved_lists: List[List[str]],
    relevant_sets: List[Union[List[str], Set[str]]],
    k: int = 5,
) -> dict[str, float]:
    """Compute all three metrics over a query set.

    Parameters
    ----------
    retrieved_lists : list of list[str]
        One ranked retrieved list per query.
    relevant_sets   : list of set[str] | list[str]
        Ground-truth relevant IDs, aligned with retrieved_lists.
    k               : int
        Rank cut-off, default 5.

    Returns
    -------
    dict with keys:
        "ndcg@k"   : mean NDCG@k across all queries
        "recall@k" : mean Recall@k across all queries
        "mrr"      : Mean Reciprocal Rank (uses same k cut-off)

    Raises
    ------
    ValueError
        If retrieved_lists and relevant_sets have different lengths.

    Examples
    --------
    >>> retrieved = [["a", "b", "x"], ["b", "a", "y"]]
    >>> relevant  = [{"a", "b"},      {"a", "b"}     ]
    >>> evaluate_ranking(retrieved, relevant, k=3)
    {'ndcg@5': ..., 'recall@5': ..., 'mrr': ...}
    """
    if len(retrieved_lists) != len(relevant_sets):
        raise ValueError(
            f"retrieved_lists and relevant_sets must have the same length, "
            f"got {len(retrieved_lists)} vs {len(relevant_sets)}"
        )

    n = len(retrieved_lists)

    if n == 0:
        return {f"ndcg@{k}": 0.0, f"recall@{k}": 0.0, "mrr": 0.0}

    ndcg_scores   = [ndcg_at_k(r, g, k)   for r, g in zip(retrieved_lists, relevant_sets)]
    recall_scores = [recall_at_k(r, g, k) for r, g in zip(retrieved_lists, relevant_sets)]

    mrr_pairs = list(zip(retrieved_lists, relevant_sets))

    return {
        f"ndcg@{k}":   sum(ndcg_scores)   / n,
        f"recall@{k}": sum(recall_scores) / n,
        "mrr":         mean_reciprocal_rank(mrr_pairs, k),
    }
