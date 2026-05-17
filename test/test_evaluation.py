"""Tests for src/evaluation.py — CSAI415 D1.

Place at tests/test_evaluation.py.
Run with: pytest tests/test_evaluation.py -v
"""

import pytest
import numpy as np

from src.automl_utils import RetrieverConfig
from src.data_utils import build_corpus, Query
from src.evaluation import evaluate_retriever, evaluate_retriever_per_query
from src.retriever import HybridKNNRetriever


# ---------------------------------------------------------------------------
# shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def corpus():
    chunks, queries = build_corpus(n_papers=16, chunks_per_paper=5, seed=42)
    return chunks, queries


@pytest.fixture(scope="module")
def fitted_retriever(corpus):
    chunks, _ = corpus
    config = RetrieverConfig(k=5, metric="cosine", svd_dim=16, normalize=True, alpha=0.5)
    return HybridKNNRetriever(config).fit(chunks)


@pytest.fixture(scope="module")
def gold_queries(corpus):
    _, queries = corpus
    return queries


# ---------------------------------------------------------------------------
# evaluate_retriever — return shape and key names
# ---------------------------------------------------------------------------

class TestEvaluateRetrieverShape:

    def test_returns_dict(self, fitted_retriever, gold_queries):
        result = evaluate_retriever(fitted_retriever, gold_queries, k=5, repeats=2)
        assert isinstance(result, dict)

    def test_contains_expected_keys_for_k5(self, fitted_retriever, gold_queries):
        result = evaluate_retriever(fitted_retriever, gold_queries, k=5, repeats=2)
        assert "ndcg@5"   in result
        assert "recall@5" in result
        assert "mrr"      in result
        assert "p95_ms"   in result
        assert "mean_ms"  in result

    def test_k_suffix_matches_argument(self, fitted_retriever, gold_queries):
        result = evaluate_retriever(fitted_retriever, gold_queries, k=3, repeats=2)
        assert "ndcg@3"   in result
        assert "recall@3" in result
        # mrr and latency keys are k-independent
        assert "mrr"     in result
        assert "p95_ms"  in result

    def test_quality_metrics_in_unit_interval(self, fitted_retriever, gold_queries):
        result = evaluate_retriever(fitted_retriever, gold_queries, k=5, repeats=2)
        for key in ("ndcg@5", "recall@5", "mrr"):
            assert 0.0 <= result[key] <= 1.0, f"{key} = {result[key]} out of [0, 1]"

    def test_latency_metrics_are_positive(self, fitted_retriever, gold_queries):
        result = evaluate_retriever(fitted_retriever, gold_queries, k=5, repeats=2)
        assert result["p95_ms"]  > 0.0
        assert result["mean_ms"] > 0.0

    def test_p95_geq_mean(self, fitted_retriever, gold_queries):
        # p95 must always be >= mean for any non-degenerate distribution.
        result = evaluate_retriever(fitted_retriever, gold_queries, k=5, repeats=5)
        assert result["p95_ms"] >= result["mean_ms"]


# ---------------------------------------------------------------------------
# evaluate_retriever — latency sample count
# ---------------------------------------------------------------------------

class TestLatencySampleCount:

    def test_p95_uses_all_repeats(self, fitted_retriever, gold_queries):
        """With repeats=5 and 40 queries, we get 200 timing samples.

        We cannot directly inspect the internal timings array, but we can
        verify that the p95 is *stable* — running the same evaluation twice
        should give identical quality scores (deterministic retriever) and
        close latency estimates.
        """
        r1 = evaluate_retriever(fitted_retriever, gold_queries, k=5, repeats=5)
        r2 = evaluate_retriever(fitted_retriever, gold_queries, k=5, repeats=5)

        # Quality is deterministic — must be bit-for-bit identical.
        assert r1["ndcg@5"]   == r2["ndcg@5"]
        assert r1["recall@5"] == r2["recall@5"]
        assert r1["mrr"]      == r2["mrr"]

        # Latency varies with OS scheduling; allow a generous 20ms tolerance.
        assert abs(r1["p95_ms"] - r2["p95_ms"]) < 20.0


# ---------------------------------------------------------------------------
# evaluate_retriever — input validation
# ---------------------------------------------------------------------------

class TestEvaluateRetrieverValidation:

    def test_empty_queries_raises(self, fitted_retriever):
        with pytest.raises(ValueError, match="non-empty"):
            evaluate_retriever(fitted_retriever, [], k=5, repeats=2)

    def test_repeats_zero_raises(self, fitted_retriever, gold_queries):
        with pytest.raises(ValueError, match="repeats"):
            evaluate_retriever(fitted_retriever, gold_queries, k=5, repeats=0)

    def test_unfitted_retriever_raises(self, gold_queries):
        config = RetrieverConfig(k=5, metric="cosine", svd_dim=16, normalize=True, alpha=0.5)
        unfitted = HybridKNNRetriever(config)
        with pytest.raises(RuntimeError):
            evaluate_retriever(unfitted, gold_queries, k=5, repeats=1)


# ---------------------------------------------------------------------------
# evaluate_retriever — bm25_fn integration
# ---------------------------------------------------------------------------

class TestBM25Integration:

    def _make_oracle_bm25(self, queries):
        """BM25 oracle: returns a high score for all ground-truth relevant chunks."""
        score_map = {}
        for q in queries:
            score_map[q.query_id] = {cid: 10.0 for cid in q.relevant_chunk_ids}
        return score_map

    def test_hybrid_mode_runs_without_error(self, fitted_retriever, gold_queries):
        score_map = self._make_oracle_bm25(gold_queries)

        def bm25_fn(query_text):
            # Match query by text — proxy for real BM25 in tests.
            for q in gold_queries:
                if q.query_text == query_text:
                    return score_map.get(q.query_id, {})
            return {}

        result = evaluate_retriever(
            fitted_retriever, gold_queries, k=5, repeats=2, bm25_fn=bm25_fn
        )
        assert "ndcg@5" in result
        assert 0.0 <= result["ndcg@5"] <= 1.0

    def test_rrf_mode_runs_without_error(self, fitted_retriever, gold_queries):
        def bm25_fn(_query_text):
            return {}   # empty BM25 — RRF degrades gracefully to dense-only

        result = evaluate_retriever(
            fitted_retriever, gold_queries, k=5, repeats=2,
            bm25_fn=bm25_fn, rrf_k=60,
        )
        assert "ndcg@5" in result


# ---------------------------------------------------------------------------
# evaluate_retriever_per_query
# ---------------------------------------------------------------------------

class TestEvaluateRetrieverPerQuery:

    def test_returns_one_row_per_query(self, fitted_retriever, gold_queries):
        rows = evaluate_retriever_per_query(
            fitted_retriever, gold_queries, k=5, repeats=2
        )
        assert len(rows) == len(gold_queries)

    def test_row_contains_expected_keys(self, fitted_retriever, gold_queries):
        rows = evaluate_retriever_per_query(
            fitted_retriever, gold_queries, k=5, repeats=2
        )
        expected_keys = {
            "query_id", "topic_id", "query_text", "query_type",
            "retrieved_ids", "relevant_ids",
            "ndcg", "recall", "reciprocal_rank", "mean_ms",
        }
        assert expected_keys == set(rows[0].keys())

    def test_per_query_ndcg_in_unit_interval(self, fitted_retriever, gold_queries):
        rows = evaluate_retriever_per_query(
            fitted_retriever, gold_queries, k=5, repeats=2
        )
        for row in rows:
            assert 0.0 <= row["ndcg"] <= 1.0

    def test_mean_of_per_query_ndcg_matches_aggregate(self, fitted_retriever, gold_queries):
        """Per-query mean NDCG should equal the aggregate evaluate_retriever value."""
        per_query = evaluate_retriever_per_query(
            fitted_retriever, gold_queries, k=5, repeats=2
        )
        aggregate = evaluate_retriever(
            fitted_retriever, gold_queries, k=5, repeats=2
        )
        per_query_mean = np.mean([r["ndcg"] for r in per_query])
        assert per_query_mean == pytest.approx(aggregate["ndcg@5"], abs=1e-9)

    def test_retrieved_ids_length_leq_k(self, fitted_retriever, gold_queries):
        rows = evaluate_retriever_per_query(
            fitted_retriever, gold_queries, k=5, repeats=2
        )
        for row in rows:
            assert len(row["retrieved_ids"]) <= 5

    def test_per_query_latency_positive(self, fitted_retriever, gold_queries):
        rows = evaluate_retriever_per_query(
            fitted_retriever, gold_queries, k=5, repeats=3
        )
        for row in rows:
            assert row["mean_ms"] > 0.0
