"""retriever_bridge.py — Unified retriever interface for CSAI415 D2 / D3.

FIX #2: D1 produced an AutoML-tuned RetrieverConfig (k, metric, svd_dim,
normalize, alpha) for a TF-IDF/SVD/sklearn retriever (HybridKNNRetriever).
D2 wrote a brand new HybridRetriever using bge embeddings + Qdrant + BM25,
with no link back to D1.  Neither deliverable defined a shared interface.

This module fixes that by:

  1.  Defining a RetrieverProtocol that both D1 and D2 retrievers satisfy.
  2.  Providing ProductionRetriever — a thin wrapper around D2's HybridRetriever
      that accepts D1's alpha convention and the AutoML best config.
  3.  Providing a factory function get_retriever() that D3's GraphRAG executor
      can call without knowing which underlying retriever it gets.

Why D1's HybridKNNRetriever is NOT used in production
------------------------------------------------------
D1's retriever uses TF-IDF + TruncatedSVD (LSA) as the dense side, which is
appropriate for the synthetic 400-chunk corpus but produces much weaker
embeddings than bge-small-en on real scientific text.  The AutoML study in D1
found the best config for that setup; those hyperparameters (k, svd_dim) do
not directly transfer to a Qdrant/bge stack.

What DOES transfer from D1
--------------------------
  - alpha:  the BM25 fusion weight.  D1's AdaptiveAlphaTable tracks per-topic
            alpha; ProductionRetriever.search() accepts it directly.
  - The evaluation harness (evaluate_retriever) and metrics (ndcg_at_k, etc.)
    are reusable as-is for D3's ablation table.

D3 usage
--------
    from retriever_bridge import get_retriever

    retriever = get_retriever()                 # ProductionRetriever (D2 stack)

    # With D1 AdaptiveAlphaTable:
    from src.online_learner import AdaptiveAlphaTable
    alpha_table = AdaptiveAlphaTable(topic_labels, default_alpha=0.4)
    alpha = alpha_table.get_alpha(predicted_topic)
    results = retriever.search(query, top_k=5, alpha=alpha)

    # Results are dicts with: chunk_id, doc_id, title, authors, year, venue,
    # page_start, page_end, text, citation, bm25_score, dense_score, hybrid_score
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Shared protocol — both D1 and D2 retrievers satisfy this
# ---------------------------------------------------------------------------

@runtime_checkable
class RetrieverProtocol(Protocol):
    """Minimal interface any retriever must satisfy for D3 GraphRAG compatibility."""

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Return top-k results as dicts with at least: chunk_id, text, citation."""
        ...

    def reload_chunks(self) -> None:
        """Refresh the retriever's internal index after new data is ingested."""
        ...


# ---------------------------------------------------------------------------
# ProductionRetriever — wraps D2's HybridRetriever with the unified interface
# ---------------------------------------------------------------------------

class ProductionRetriever:
    """Thin wrapper around D2's HybridRetriever that implements RetrieverProtocol.

    Accepts D1's single-alpha convention so AdaptiveAlphaTable output can be
    passed directly without any conversion at the call site.

    Parameters
    ----------
    default_alpha : float
        BM25 weight used when alpha is not supplied per-query.
        Set this to the best alpha found by D1's AutoML study (0.00 on the
        synthetic corpus; expected ~0.3-0.5 on real arXiv PDFs after D2 eval).
    """

    def __init__(self, default_alpha: float = 0.4) -> None:
        from retriever import HybridRetriever
        self._retriever = HybridRetriever()
        self.default_alpha = default_alpha

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Search using D2's hybrid BM25 + dense stack.

        Parameters
        ----------
        query : str
            Raw query text.
        top_k : int
            Number of results to return.
        alpha : float | None
            BM25 weight in [0, 1].  Follows D1's AdaptiveAlphaTable convention:
              alpha=1.0  → pure BM25
              alpha=0.0  → pure dense
              alpha=None → use self.default_alpha
        """
        effective_alpha = alpha if alpha is not None else self.default_alpha
        return self._retriever.search(
            query=query,
            top_k=top_k,
            alpha=effective_alpha,
        )

    def reload_chunks(self) -> None:
        """Reload MongoDB chunks and rebuild BM25 index (called after /ingest)."""
        self._retriever.reload_chunks()

    @property
    def chunks(self) -> list[dict]:
        """Direct access to loaded chunks — used by /stats endpoint."""
        return self._retriever.chunks


# ---------------------------------------------------------------------------
# D1RetrieverAdapter — wraps D1's HybridKNNRetriever to satisfy RetrieverProtocol
#
# Use this ONLY for ablation / offline evaluation, not in production.
# D1's retriever needs a fitted corpus of D1 Chunk objects, which do not
# exist at runtime in D2/D3 (the real corpus uses ChunkRecord/MongoDB).
# ---------------------------------------------------------------------------

class D1RetrieverAdapter:
    """Adapts D1's HybridKNNRetriever to RetrieverProtocol for ablation testing.

    Example (ablation in D3):
        from retriever_bridge import D1RetrieverAdapter
        from src.automl_utils import RetrieverConfig
        from src.data_utils import build_corpus

        best_config = RetrieverConfig(k=15, metric="cosine", svd_dim=96,
                                      normalize=False, alpha=0.0)
        chunks, _ = build_corpus()
        adapter = D1RetrieverAdapter(best_config, chunks)
        results  = adapter.search("attention mechanism", top_k=5)
    """

    def __init__(self, config: Any, chunks: list) -> None:
        import sys, os
        # D1 source lives one level up (assumes project layout:
        #   project/
        #     src/           <- D1 modules
        #     retriever.py   <- D2 (this file's directory)
        d1_src = os.path.join(os.path.dirname(__file__), "..", "src")
        if os.path.isdir(d1_src) and d1_src not in sys.path:
            sys.path.insert(0, d1_src)

        from retriever import HybridKNNRetriever as D1Retriever  # type: ignore[import]
        self._retriever = D1Retriever(config).fit(chunks)
        self._chunks = {c.chunk_id: c for c in chunks}
        self._config = config

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Run D1 retrieval and normalise results to the shared dict format."""
        # Override alpha in config if supplied
        if alpha is not None:
            from dataclasses import replace
            cfg = replace(self._config, alpha=alpha)
            import sys, os
            d1_src = os.path.join(os.path.dirname(__file__), "..", "src")
            if os.path.isdir(d1_src) and d1_src not in sys.path:
                sys.path.insert(0, d1_src)
            from retriever import HybridKNNRetriever as D1Retriever  # type: ignore
            retriever = D1Retriever(cfg).fit(list(self._chunks.values()))
        else:
            retriever = self._retriever

        hits = retriever.search(query, top_k=top_k)

        results = []
        for hit in hits:
            chunk = hit.chunk
            results.append({
                "chunk_id":    chunk.chunk_id,
                "doc_id":      chunk.paper_id,
                "title":       f"Synthetic paper {chunk.paper_id}",
                "authors":     "",
                "year":        None,
                "venue":       "",
                "page_start":  chunk.page,
                "page_end":    chunk.page,
                "text":        chunk.text,
                "citation":    f"Synthetic paper {chunk.paper_id}, page {chunk.page}",
                "bm25_score":  getattr(hit, "bm25_score_norm", 0.0) or 0.0,
                "dense_score": hit.dense_score,
                "hybrid_score": hit.fused_score if hit.fused_score is not None else hit.dense_score_norm,
            })
        return results

    def reload_chunks(self) -> None:
        """No-op for D1 retriever — corpus is fixed at construction time."""
        pass


# ---------------------------------------------------------------------------
# Factory — what D3's GraphRAG executor should call
# ---------------------------------------------------------------------------

def get_retriever(
    mode: str = "production",
    default_alpha: float = 0.4,
    d1_config: Any = None,
    d1_chunks: list | None = None,
) -> RetrieverProtocol:
    """Return a retriever that satisfies RetrieverProtocol.

    Parameters
    ----------
    mode : str
        "production"  → ProductionRetriever (D2 bge + Qdrant + BM25)  [default]
        "d1"          → D1RetrieverAdapter  (TF-IDF/SVD, ablation only)
    default_alpha : float
        Default BM25 weight when alpha is not supplied per-query.
    d1_config : RetrieverConfig | None
        Required when mode="d1".
    d1_chunks : list[D1Chunk] | None
        Required when mode="d1".

    Returns
    -------
    RetrieverProtocol
        A retriever with .search(query, top_k, alpha) and .reload_chunks().
    """
    if mode == "production":
        return ProductionRetriever(default_alpha=default_alpha)
    elif mode == "d1":
        if d1_config is None or d1_chunks is None:
            raise ValueError("d1_config and d1_chunks are required for mode='d1'")
        return D1RetrieverAdapter(d1_config, d1_chunks)
    else:
        raise ValueError(f"Unknown retriever mode: {mode!r}. Choose 'production' or 'd1'.")
