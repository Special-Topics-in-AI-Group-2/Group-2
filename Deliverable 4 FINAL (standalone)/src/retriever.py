"""Hybrid (dense + lexical) retriever for CSAI415 D1.

Dense pipeline
--------------
Chunk.text
    → TfidfVectorizer          (sparse term-frequency matrix)
    → TruncatedSVD             (LSA projection to svd_dim dimensions)
    → L2 normalisation         (conditional on RetrieverConfig.normalize)
    → NearestNeighbors index   (cosine or euclidean, brute-force for D1 scale)

Weighted Score Fusion (WSF)
---------------------------
BM25 scores are computed externally and passed into search() as a
dict[chunk_id → raw_score].  Fusion is applied when bm25_scores is provided:

    fused = alpha * norm(bm25) + (1 - alpha) * norm(dense)

where norm() is per-query min-max normalisation applied independently to each
modality.  Candidates present in only one modality receive a score of 0.0 for
the missing modality after normalisation (i.e. "worst in the retrieved set").

The candidate pool is the union of the dense top-k and the BM25 top-k, so the
final result list may contain more than top_k items before the final re-rank
truncation.

Note on metric + normalize interaction
---------------------------------------
On L2-normalised vectors cosine and euclidean induce identical rankings because:

    ||a - b||^2 = 2 - 2·cos(a, b)   when ||a|| = ||b|| = 1

RetrieverConfig.normalize=True + metric="euclidean" is therefore a dead
hyperparameter.  Both branches are supported so Optuna can discover this
empirically, but consider collapsing to a single combo in your search space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from src.automl_utils import RetrieverConfig
from src.data_utils import Chunk


@dataclass
class HybridResult:
    """Single result returned by HybridKNNRetriever.search().

    Attributes
    ----------
    chunk : Chunk
        The matched chunk object.
    dense_score : float
        Raw cosine similarity (or negative euclidean distance).
        Higher is always more relevant regardless of metric.
    dense_score_norm : float
        Min-max normalised dense score for this query, in [0, 1].
    bm25_score_norm : float | None
        Min-max normalised BM25 score for this query, in [0, 1].
        None when search() is called without bm25_scores (dense-only mode).
    fused_score : float | None
        alpha * bm25_score_norm + (1 - alpha) * dense_score_norm.
        None in dense-only mode.  Use this for final ranking when present.
    rank : int
        1-based rank in the returned list (1 = most relevant).
        Ranked by fused_score when available, dense_score_norm otherwise.
    """

    chunk: Chunk
    dense_score: float
    dense_score_norm: float
    bm25_score_norm: float | None
    fused_score: float | None
    rank: int


# Backward-compatible alias so existing code referencing DenseResult still works.
DenseResult = HybridResult


def _minmax_norm(scores: np.ndarray) -> np.ndarray:
    """Per-query min-max normalisation into [0, 1].

    Three cases:

    hi == lo == 0  →  zeros: the modality has *no signal* for this query
                      (e.g. all BM25 scores are 0.0 because no query term
                      matched any chunk).  Returning zeros means this modality
                      contributes nothing to fusion, so the fused score
                      gracefully degrades to (1 - alpha) * dense_norm.

    hi == lo != 0  →  ones: every candidate is equally relevant according to
                      this modality (uniform signal).  Returning ones gives
                      each candidate full credit from this side.

    hi > lo        →  standard min-max: maps [lo, hi] → [0, 1].
    """
    lo, hi = scores.min(), scores.max()
    if hi == lo:
        return (
            np.zeros_like(scores, dtype=float)
            if hi == 0.0
            else np.ones_like(scores, dtype=float)
        )
    return (scores - lo) / (hi - lo)

def _rrf_fuse(
    ordered_ids: list[str],
    dense_ranks: dict[str, int],
    bm25_ranks: dict[str, int],
    k_rrf: int = 60,
) -> np.ndarray:
    """Reciprocal Rank Fusion across two ranked lists.

    For each candidate in ordered_ids:

        score(d) = 1 / (k_rrf + rank_dense(d))
                 + 1 / (k_rrf + rank_bm25(d))

    Candidates absent from a modality receive a penalty rank of
    (pool_size + 1), contributing a near-zero positive term rather
    than zero.  This is preferable to zero because it preserves the
    semantic that "seen but ranked last" beats "never retrieved" —
    the same principle as WSF's union-pool zero-score floor.

    Parameters
    ----------
    ordered_ids : list[str]
        Stable-ordered union of all candidate chunk IDs.
    dense_ranks : dict[str, int]
        chunk_id → 1-based rank from the dense retriever (1 = best).
    bm25_ranks : dict[str, int]
        chunk_id → 1-based rank from BM25 (1 = best).
    k_rrf : int
        Smoothing constant.  k_rrf=60 is the standard value from
        Cormack et al. (2009) "Reciprocal Rank Fusion outperforms
        Condorcet and individual rank learning methods".

    Returns
    -------
    np.ndarray, shape (len(ordered_ids),)
        RRF scores, higher = more relevant.
    """
    penalty_rank = len(ordered_ids) + 1
    scores = np.array([
        1.0 / (k_rrf + dense_ranks.get(cid, penalty_rank))
        + 1.0 / (k_rrf + bm25_ranks.get(cid, penalty_rank))
        for cid in ordered_ids
    ])
    return scores




class HybridKNNRetriever:
    """TF-IDF + TruncatedSVD + NearestNeighbors retriever with optional WSF fusion.

    Parameters
    ----------
    config : RetrieverConfig
        Hyperparameter bundle from automl_utils.  All five fields are used:
        k, metric, svd_dim, normalize (dense pipeline) and alpha (WSF fusion).

    Usage — dense only
    ------------------
    >>> retriever = HybridKNNRetriever(config)
    >>> retriever.fit(chunks)
    >>> results = retriever.search(query_text, top_k=5)

    Usage — hybrid (WSF)
    --------------------
    >>> bm25_scores = {"chunk_0001": 4.2, "chunk_0007": 2.1, ...}
    >>> results = retriever.search(query_text, top_k=5, bm25_scores=bm25_scores)
    """

    def __init__(self, config: RetrieverConfig) -> None:
        self.config = config

        # --- pipeline components (populated by fit) --------------------------
        self._tfidf: TfidfVectorizer | None = None
        self._svd: TruncatedSVD | None = None
        self._index: NearestNeighbors | None = None

        # --- corpus store (populated by fit) ---------------------------------
        self._chunks: List[Chunk] = []
        self._matrix: np.ndarray | None = None           # shape (n_chunks, svd_dim)
        self._chunk_id_index: dict[str, Chunk] = {}      # chunk_id → Chunk

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------

    def fit(self, chunks: List[Chunk]) -> "HybridKNNRetriever":
        """Build TF-IDF → SVD → (optional L2 norm) → NearestNeighbors index.

        Parameters
        ----------
        chunks : list[Chunk]
            All corpus chunks.  Order is preserved so index position maps
            directly to chunks[i].

        Returns
        -------
        self
            Allows chaining: retriever.fit(chunks).search(query, k).

        Notes
        -----
        * TfidfVectorizer uses sublinear_tf=True (log-normalised term freq)
          which compresses the dynamic range of common terms — a small but
          meaningful improvement over raw counts for short chunks.
        * TruncatedSVD n_components must be < vocabulary size.  The guard
          below silently caps svd_dim if the corpus vocabulary is tiny (e.g.
          in unit tests with a handful of synthetic chunks).
        * random_state=42 on TruncatedSVD ensures the SVD decomposition is
          reproducible across runs regardless of numpy's global RNG state.
        """

        if not chunks:
            raise ValueError("chunks must be non-empty.")

        self._chunks = list(chunks)
        texts = [c.text for c in self._chunks]

        # ── 1. TF-IDF ────────────────────────────────────────────────────────
        self._tfidf = TfidfVectorizer(sublinear_tf=True)
        sparse_matrix = self._tfidf.fit_transform(texts)   # (n, vocab)

        # ── 2. TruncatedSVD (LSA) ────────────────────────────────────────────
        vocab_size = sparse_matrix.shape[1]
        safe_svd_dim = min(self.config.svd_dim, vocab_size - 1)

        self._svd = TruncatedSVD(
            n_components=safe_svd_dim,
            random_state=42,
        )
        dense_matrix = self._svd.fit_transform(sparse_matrix)  # (n, svd_dim)

        # ── 3. Optional L2 normalisation ─────────────────────────────────────
        if self.config.normalize:
            dense_matrix = normalize(dense_matrix, norm="l2")

        self._matrix = dense_matrix

        # ── 4. NearestNeighbors index ─────────────────────────────────────────
        # algorithm="brute" is correct here: at D1 scale (≤ 400 chunks) tree
        # structures (ball_tree, kd_tree) add overhead without benefit.
        # For D2+ with thousands of real chunks, switch to Qdrant.
        nn_metric = (
            "cosine"
            if self.config.metric == "cosine"
            else "euclidean"
        )
        self._index = NearestNeighbors(
            n_neighbors=self.config.k,
            metric=nn_metric,
            algorithm="brute",
        )
        self._index.fit(self._matrix)

        # ── 5. chunk_id → Chunk lookup (for BM25-only union candidates) ───────
        self._chunk_id_index: dict[str, Chunk] = {
            c.chunk_id: c for c in self._chunks
        }

        return self

    # ------------------------------------------------------------------
    # search
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int | None = None,
        bm25_scores: dict[str, float] | None = None,
        rrf_k: int | None = None,
    ) -> List[HybridResult]:
        """Return the top-k most relevant chunks for a query string.

        Parameters
        ----------
        query : str
            Raw query text.  The fitted TF-IDF → SVD → (norm) pipeline is
            applied via transform() — corpus is never refitted.
        top_k : int | None
            Number of results to return.  Defaults to config.k.
        bm25_scores : dict[str, float] | None
            Optional mapping of chunk_id → raw BM25 score for the same query,
            computed externally (e.g. rank_bm25 or Elasticsearch BM25).
            When provided, WSF or RRF fusion is applied (see rrf_k).
            When None, the method operates in dense-only mode and
            HybridResult.bm25_score_norm / .fused_score are both None.
        rrf_k : int | None
            When provided alongside bm25_scores, Reciprocal Rank Fusion is
            used instead of WSF.  The standard value is 60 (Cormack 2009).
            RRF ignores RetrieverConfig.alpha entirely — there is no per-
            modality weight, only rank-based combination.
            When None (default), WSF with min-max normalisation is used.

        Returns
        -------
        list[HybridResult]
            Sorted descending by fused_score (hybrid mode) or dense_score_norm
            (dense-only mode).  Always length min(top_k, n_candidates).

        Raises
        ------
        RuntimeError
            If called before fit().
        """

        self._require_fitted()

        k = top_k if top_k is not None else self.config.k

        if k > self._index.n_neighbors:
            self._index.set_params(n_neighbors=k)

        # ── 1. Dense retrieval ────────────────────────────────────────────────
        query_vec = self._encode_query(query)           # (1, svd_dim)
        distances, indices = self._index.kneighbors(query_vec, n_neighbors=k)
        distances = distances[0]
        indices   = indices[0]

        if self.config.metric == "cosine":
            raw_dense = 1.0 - distances          # cosine similarity ∈ [-1, 1]
        else:
            raw_dense = -distances               # negated euclidean, higher = closer

        # Map chunk_id → (chunk_object, raw_dense_score) for union step
        dense_map: dict[str, tuple[Chunk, float]] = {
            self._chunks[idx].chunk_id: (self._chunks[idx], float(score))
            for idx, score in zip(indices, raw_dense)
        }

        # ── 2. Dense-only mode ────────────────────────────────────────────────
        if bm25_scores is None:
            dense_scores_arr = np.array([v for _, v in dense_map.values()])
            dense_norm_arr   = _minmax_norm(dense_scores_arr)

            results = [
                HybridResult(
                    chunk=chunk,
                    dense_score=raw,
                    dense_score_norm=float(norm),
                    bm25_score_norm=None,
                    fused_score=None,
                    rank=0,             # assigned below
                )
                for (chunk, raw), norm in zip(dense_map.values(), dense_norm_arr)
            ]
            results.sort(key=lambda r: r.dense_score_norm, reverse=True)
            results = results[:k]
            for i, r in enumerate(results, start=1):
                r.rank = i
            return results

        # ── 3. Hybrid mode — build union candidate pool ───────────────────────
        #
        # Union = dense_map keys ∪ bm25_scores keys.
        # Chunks in BM25 but not in dense_map need their Chunk objects looked up.
        # Use the pre-built _chunk_id_index for O(1) lookup.
        #
        all_ids: set[str] = set(dense_map) | set(bm25_scores)

        raw_dense_union: dict[str, float] = {}
        raw_bm25_union:  dict[str, float] = {}
        chunks_union:    dict[str, Chunk]  = {}

        for cid in all_ids:
            if cid in dense_map:
                chunks_union[cid]    = dense_map[cid][0]
                raw_dense_union[cid] = dense_map[cid][1]
            else:
                # BM25-only candidate: fetch Chunk from index, dense score = 0
                # before normalisation (treated as worst dense candidate).
                looked_up = self._chunk_id_index.get(cid)
                if looked_up is None:
                    # BM25 returned a chunk_id not in the fitted corpus — skip.
                    continue
                chunks_union[cid]    = looked_up
                raw_dense_union[cid] = 0.0

            raw_bm25_union[cid] = bm25_scores.get(cid, 0.0)

        ordered_ids = list(chunks_union)          # stable iteration order

        # ── 4a. RRF fusion ────────────────────────────────────────────────────
        if rrf_k is not None:
            # Convert raw scores → 1-based ranks (rank 1 = highest score).
            # sorted(..., reverse=True) gives descending score order; enumerate
            # from 1 gives the rank.  Ties share the better rank (first-seen
            # wins, which is deterministic given stable sort).
            dense_sorted = sorted(
                dense_map.keys(), key=lambda cid: raw_dense_union.get(cid, 0.0), reverse=True
            )
            bm25_sorted  = sorted(
                bm25_scores.keys(), key=bm25_scores.get, reverse=True
            )

            dense_ranks: dict[str, int] = {cid: r for r, cid in enumerate(dense_sorted, 1)}
            bm25_ranks:  dict[str, int] = {cid: r for r, cid in enumerate(bm25_sorted,  1)}

            fused = _rrf_fuse(ordered_ids, dense_ranks, bm25_ranks, k_rrf=rrf_k)

            # RRF produces raw scores — normalise to [0, 1] for HybridResult
            # consistency so callers can compare across queries.
            fused_norm = _minmax_norm(fused)

            # dense_score_norm and bm25_score_norm are still reported for
            # inspection / ablation, but they play no role in the RRF ranking.
            dense_arr = np.array([raw_dense_union[cid] for cid in ordered_ids])
            bm25_arr  = np.array([raw_bm25_union[cid]  for cid in ordered_ids])
            dense_norm = _minmax_norm(dense_arr)
            bm25_norm  = _minmax_norm(bm25_arr)

            results = [
                HybridResult(
                    chunk=chunks_union[cid],
                    dense_score=raw_dense_union[cid],
                    dense_score_norm=float(dn),
                    bm25_score_norm=float(bn),
                    fused_score=float(fs),
                    rank=0,
                )
                for cid, dn, bn, fs in zip(ordered_ids, dense_norm, bm25_norm, fused_norm)
            ]

        # ── 4b. WSF fusion ────────────────────────────────────────────────────
        else:
            dense_arr = np.array([raw_dense_union[cid] for cid in ordered_ids])
            bm25_arr  = np.array([raw_bm25_union[cid]  for cid in ordered_ids])

            dense_norm = _minmax_norm(dense_arr)
            bm25_norm  = _minmax_norm(bm25_arr)

            alpha = self.config.alpha
            fused = alpha * bm25_norm + (1.0 - alpha) * dense_norm

            results = [
                HybridResult(
                    chunk=chunks_union[cid],
                    dense_score=raw_dense_union[cid],
                    dense_score_norm=float(dn),
                    bm25_score_norm=float(bn),
                    fused_score=float(fs),
                    rank=0,
                )
                for cid, dn, bn, fs in zip(ordered_ids, dense_norm, bm25_norm, fused)
            ]

        # ── 5. Sort, truncate, assign ranks ───────────────────────────────────
        results.sort(key=lambda r: r.fused_score, reverse=True)
        results = results[:k]
        for i, r in enumerate(results, start=1):
            r.rank = i

        return results

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _encode_query(self, text: str) -> np.ndarray:
        """Apply the fitted TF-IDF → SVD → (norm) pipeline to a single string.

        Returns a (1, svd_dim) array ready for NearestNeighbors.kneighbors().
        """
        sparse = self._tfidf.transform([text])
        dense  = self._svd.transform(sparse)

        if self.config.normalize:
            dense = normalize(dense, norm="l2")

        return dense

    def _require_fitted(self) -> None:
        if self._index is None:
            raise RuntimeError(
                "HybridKNNRetriever.fit(chunks) must be called before search()."
            )

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        fitted = self._index is not None
        return (
            f"HybridKNNRetriever("
            f"k={self.config.k}, "
            f"metric={self.config.metric!r}, "
            f"svd_dim={self.config.svd_dim}, "
            f"normalize={self.config.normalize}, "
            f"fitted={fitted})"
        )
