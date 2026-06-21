"""Tests for src/metrics.py — CSAI415 D1.

Place this file at tests/test_metrics.py.
Run with: pytest tests/test_metrics.py -v
"""

import math
import pytest

from src.metrics import (
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    mean_reciprocal_rank,
    evaluate_ranking,
)


# ---------------------------------------------------------------------------
# ndcg_at_k
# ---------------------------------------------------------------------------

class TestNdcgAtK:

    def test_perfect_ranking_returns_1(self):
        # All 3 relevant docs at the top 3 positions — ideal ranking.
        retrieved = ["a", "b", "c", "d", "e"]
        relevant  = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, relevant, k=5) == pytest.approx(1.0)

    def test_zero_when_no_relevant_retrieved(self):
        retrieved = ["x", "y", "z", "w", "v"]
        relevant  = {"a", "b", "c"}
        assert ndcg_at_k(retrieved, relevant, k=5) == 0.0

    def test_lower_score_when_relevant_docs_ranked_later(self):
        # Relevant docs at the bottom vs at the top of the window.
        retrieved_good = ["a", "b", "c", "x", "y"]   # relevant first
        retrieved_poor = ["x", "y", "a", "b", "c"]   # relevant last
        relevant = {"a", "b", "c"}
        good_score = ndcg_at_k(retrieved_good, relevant, k=5)
        poor_score = ndcg_at_k(retrieved_poor, relevant, k=5)
        assert good_score > poor_score

    def test_empty_retrieved_returns_0(self):
        assert ndcg_at_k([], {"a", "b"}, k=5) == 0.0

    def test_empty_relevant_returns_0(self):
        assert ndcg_at_k(["a", "b", "c"], set(), k=5) == 0.0

    def test_accepts_list_for_relevant_ids(self):
        # relevant_ids can be a list — should not raise.
        score = ndcg_at_k(["a", "b", "c"], ["a", "b", "c"], k=5)
        assert score == pytest.approx(1.0)

    def test_single_relevant_doc_at_rank_1(self):
        # DCG = 1/log2(2) = 1.0; IDCG = 1.0 → NDCG = 1.0
        assert ndcg_at_k(["a", "x", "y"], {"a"}, k=3) == pytest.approx(1.0)

    def test_single_relevant_doc_at_rank_2(self):
        # DCG = 1/log2(3); IDCG = 1/log2(2) = 1.0
        expected = (1.0 / math.log2(3)) / 1.0
        assert ndcg_at_k(["x", "a", "y"], {"a"}, k=3) == pytest.approx(expected)

    def test_single_relevant_doc_outside_k_returns_0(self):
        assert ndcg_at_k(["x", "y", "z", "a"], {"a"}, k=3) == 0.0

    def test_k_larger_than_retrieved_list(self):
        # Should not crash; treats missing positions as non-relevant.
        score = ndcg_at_k(["a", "b"], {"a", "b", "c"}, k=10)
        assert 0.0 < score < 1.0

    def test_dcg_denominator_convention(self):
        """Verify the log2(i+1) denominator for ranks 1–3 explicitly."""
        # One relevant doc at each rank, checked individually.
        relevant = {"target"}

        score_rank1 = ndcg_at_k(["target", "x", "y"], relevant, k=3)
        score_rank2 = ndcg_at_k(["x", "target", "y"], relevant, k=3)
        score_rank3 = ndcg_at_k(["x", "y", "target"], relevant, k=3)

        # IDCG for 1 relevant doc = 1/log2(2) = 1.0 in all cases.
        # DCG at rank i = 1/log2(i+1), so NDCG = DCG/1.0 = 1/log2(i+1).
        assert score_rank1 == pytest.approx(1.0 / math.log2(2))   # = 1.0
        assert score_rank2 == pytest.approx(1.0 / math.log2(3))   # ≈ 0.631
        assert score_rank3 == pytest.approx(1.0 / math.log2(4))   # = 0.5

        # Rank 1 score must be strictly greater than rank 2, etc.
        assert score_rank1 > score_rank2 > score_rank3


# ---------------------------------------------------------------------------
# recall_at_k
# ---------------------------------------------------------------------------

class TestRecallAtK:

    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c", "x", "y"], {"a", "b", "c"}, k=5) == pytest.approx(1.0)

    def test_partial_recall(self):
        # 2 of 3 relevant docs in top 5.
        assert recall_at_k(["a", "b", "x", "y", "z"], {"a", "b", "c"}, k=5) == pytest.approx(2/3)

    def test_zero_recall(self):
        assert recall_at_k(["x", "y", "z"], {"a", "b", "c"}, k=5) == 0.0

    def test_recall_is_position_blind(self):
        # Same docs retrieved in different orders → same recall.
        r1 = recall_at_k(["a", "b", "c", "x", "y"], {"a", "b", "c"}, k=5)
        r2 = recall_at_k(["x", "y", "a", "b", "c"], {"a", "b", "c"}, k=5)
        assert r1 == r2

    def test_cutoff_is_respected(self):
        # Third relevant doc is at position 4 — outside k=3.
        assert recall_at_k(["a", "b", "x", "c", "y"], {"a", "b", "c"}, k=3) == pytest.approx(2/3)

    def test_empty_retrieved_returns_0(self):
        assert recall_at_k([], {"a", "b"}, k=5) == 0.0

    def test_empty_relevant_returns_0(self):
        assert recall_at_k(["a", "b"], set(), k=5) == 0.0

    def test_accepts_list_for_relevant_ids(self):
        score = recall_at_k(["a", "b", "c"], ["a", "b", "c"], k=5)
        assert score == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# reciprocal_rank + mean_reciprocal_rank
# ---------------------------------------------------------------------------

class TestReciprocalRank:

    def test_first_relevant_at_rank_1(self):
        assert reciprocal_rank(["a", "b", "c"], {"a"}, k=5) == pytest.approx(1.0)

    def test_first_relevant_at_rank_2(self):
        assert reciprocal_rank(["x", "a", "c"], {"a"}, k=5) == pytest.approx(0.5)

    def test_first_relevant_at_rank_5(self):
        assert reciprocal_rank(["x", "y", "z", "w", "a"], {"a"}, k=5) == pytest.approx(0.2)

    def test_no_relevant_in_top_k(self):
        assert reciprocal_rank(["x", "y", "z"], {"a"}, k=3) == 0.0

    def test_returns_rank_of_first_not_second(self):
        # Both "a" and "b" are relevant; RR should reflect rank of "a" (rank 3).
        rr = reciprocal_rank(["x", "y", "a", "b", "z"], {"a", "b"}, k=5)
        assert rr == pytest.approx(1.0 / 3)


class TestMeanReciprocalRank:

    def test_basic_mrr(self):
        pairs = [
            (["a", "b", "c"], {"a"}),   # RR = 1.0
            (["x", "a", "c"], {"a"}),   # RR = 0.5
            (["x", "y", "z"], {"a"}),   # RR = 0.0
        ]
        assert mean_reciprocal_rank(pairs, k=5) == pytest.approx((1.0 + 0.5 + 0.0) / 3)

    def test_all_hits_at_rank_1(self):
        pairs = [
            (["a", "x"], {"a"}),
            (["b", "x"], {"b"}),
        ]
        assert mean_reciprocal_rank(pairs, k=5) == pytest.approx(1.0)

    def test_empty_query_list_returns_0(self):
        assert mean_reciprocal_rank([], k=5) == 0.0


# ---------------------------------------------------------------------------
# evaluate_ranking (convenience wrapper)
# ---------------------------------------------------------------------------

class TestEvaluateRanking:

    def test_returns_all_three_keys(self):
        retrieved = [["a", "b", "c"]]
        relevant  = [{"a", "b", "c"}]
        result = evaluate_ranking(retrieved, relevant, k=5)
        assert "ndcg@5"   in result
        assert "recall@5" in result
        assert "mrr"      in result

    def test_perfect_retrieval_all_ones(self):
        retrieved = [["a", "b", "c", "x", "y"]]
        relevant  = [{"a", "b", "c"}]
        result = evaluate_ranking(retrieved, relevant, k=5)
        assert result["ndcg@5"]   == pytest.approx(1.0)
        assert result["recall@5"] == pytest.approx(1.0)
        assert result["mrr"]      == pytest.approx(1.0)

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            evaluate_ranking([["a"]], [{"a"}, {"b"}], k=5)

    def test_empty_inputs_return_zeros(self):
        result = evaluate_ranking([], [], k=5)
        assert result["ndcg@5"]   == 0.0
        assert result["recall@5"] == 0.0
        assert result["mrr"]      == 0.0

    def test_k_suffix_in_keys_matches_argument(self):
        result = evaluate_ranking([["a"]], [{"a"}], k=3)
        assert "ndcg@3"   in result
        assert "recall@3" in result
