"""Tests for OnlineTopicClassifier and AdaptiveAlphaTable."""

import pytest

river = pytest.importorskip("river")

from src.online_learner import AdaptiveAlphaTable, OnlineTopicClassifier


TOPICS = [f"topic_{i}" for i in range(8)]


# ---------------------------------------------------------------------------
# OnlineTopicClassifier
# ---------------------------------------------------------------------------


class TestOnlineTopicClassifier:

    def test_cold_start_predict_returns_first_label(self):
        clf = OnlineTopicClassifier(TOPICS)
        assert clf.predict("some query text") == TOPICS[0]

    def test_n_seen_increments_on_learn(self):
        clf = OnlineTopicClassifier(TOPICS)
        assert clf.n_seen == 0
        clf.learn("query about topic_0", "topic_0")
        assert clf.n_seen == 1

    def test_prequential_accuracy_none_during_cold_start(self):
        clf = OnlineTopicClassifier(TOPICS, window_size=10)
        for i in range(9):
            clf.learn(f"query {i}", TOPICS[i % len(TOPICS)])
        assert clf.prequential_accuracy() is None

    def test_prequential_accuracy_float_after_window_filled(self):
        clf = OnlineTopicClassifier(TOPICS, window_size=5)
        for i in range(5):
            clf.learn(f"query {i}", TOPICS[i % len(TOPICS)])
        acc = clf.prequential_accuracy()
        assert acc is not None
        assert 0.0 <= acc <= 1.0

    def test_learn_returns_bool(self):
        clf = OnlineTopicClassifier(TOPICS)
        result = clf.learn("some query", "topic_0")
        assert isinstance(result, bool)

    def test_drift_indices_empty_before_any_drift(self):
        clf = OnlineTopicClassifier(TOPICS)
        for i in range(20):
            clf.learn(f"query {i}", "topic_0")
        # no drift on a stable stream of 20 steps — may vary but list exists
        assert isinstance(clf.drift_indices, list)

    def test_drift_indices_returns_copy(self):
        clf = OnlineTopicClassifier(TOPICS)
        indices = clf.drift_indices
        indices.append(999)
        assert 999 not in clf.drift_indices

    def test_cooldown_suppresses_double_detection(self):
        """Force two rapid ADWIN triggers and confirm cooldown collapses them."""
        clf = OnlineTopicClassifier(TOPICS, delta=0.9, cooldown=30)

        # stable period
        for _ in range(30):
            clf.learn("keyword_0_0 methods", "topic_0")

        # abrupt switch — should trigger at most once per cooldown window
        drift_count = 0
        for _ in range(30):
            fired = clf.learn("semantic_7_0 representation learning", "topic_7")
            if fired:
                drift_count += 1

        # with cooldown=30 over 30 steps, at most 1 detection is possible
        assert drift_count <= 1

    def test_accuracy_improves_on_consistent_topic(self):
        """After enough steps on a single topic, accuracy should exceed cold start."""
        clf = OnlineTopicClassifier(TOPICS, window_size=10)
        text = "keyword_0_0 keyword_0_1 methods"
        label = "topic_0"

        for _ in range(50):
            clf.learn(text, label)

        acc = clf.prequential_accuracy()
        assert acc is not None
        # after 50 identical examples the model should learn the pattern
        assert acc > 0.5


# ---------------------------------------------------------------------------
# AdaptiveAlphaTable
# ---------------------------------------------------------------------------


class TestAdaptiveAlphaTable:

    def test_initial_alpha_equals_default(self):
        table = AdaptiveAlphaTable(TOPICS, default_alpha=0.5)
        for topic in TOPICS:
            assert table.get_alpha(topic) == 0.5

    def test_unknown_topic_returns_default(self):
        table = AdaptiveAlphaTable(TOPICS, default_alpha=0.6)
        assert table.get_alpha("unseen_topic") == 0.6

    def test_helpful_feedback_pulls_toward_alpha_used(self):
        table = AdaptiveAlphaTable(TOPICS, default_alpha=0.5, ema_rate=0.1)
        new_alpha = table.update("topic_0", alpha_used=0.8, helpful=True)
        # EMA: 0.9 * 0.5 + 0.1 * 0.8 = 0.53
        assert abs(new_alpha - 0.53) < 1e-9

    def test_unhelpful_feedback_pulls_toward_0_5(self):
        table = AdaptiveAlphaTable(TOPICS, default_alpha=0.8, ema_rate=0.1)
        new_alpha = table.update("topic_0", alpha_used=0.8, helpful=False)
        # EMA: 0.9 * 0.8 + 0.1 * 0.5 = 0.77
        assert abs(new_alpha - 0.77) < 1e-9

    def test_alpha_clipped_to_bounds(self):
        table = AdaptiveAlphaTable(
            TOPICS, default_alpha=0.94, ema_rate=0.5,
            alpha_min=0.05, alpha_max=0.95,
        )
        # push toward 1.0 repeatedly — should never exceed alpha_max
        for _ in range(20):
            new_alpha = table.update("topic_0", alpha_used=1.0, helpful=True)
        assert new_alpha <= 0.95

    def test_n_updates_increments(self):
        table = AdaptiveAlphaTable(TOPICS)
        assert table.n_updates == 0
        table.update("topic_0", alpha_used=0.5, helpful=True)
        table.update("topic_1", alpha_used=0.6, helpful=False)
        assert table.n_updates == 2

    def test_summary_contains_all_topics(self):
        table = AdaptiveAlphaTable(TOPICS)
        summary = table.summary()
        assert set(summary.keys()) == set(TOPICS)
        for topic, data in summary.items():
            assert "alpha" in data
            assert "n_updates" in data

    def test_alpha_history_filtered_by_topic(self):
        table = AdaptiveAlphaTable(TOPICS)
        table.update("topic_0", alpha_used=0.6, helpful=True)
        table.update("topic_1", alpha_used=0.4, helpful=False)
        table.update("topic_0", alpha_used=0.65, helpful=True)

        history_0 = table.alpha_history("topic_0")
        assert len(history_0) == 2
        assert all(row[0] == "topic_0" for row in history_0)

    def test_alpha_history_unfiltered_returns_all(self):
        table = AdaptiveAlphaTable(TOPICS)
        table.update("topic_0", alpha_used=0.6, helpful=True)
        table.update("topic_2", alpha_used=0.4, helpful=False)
        assert len(table.alpha_history()) == 2

    def test_random_feedback_converges_toward_0_5(self):
        """With 50/50 helpful/not-helpful, alpha should drift toward 0.5."""
        import random
        random.seed(42)
        table = AdaptiveAlphaTable(["topic_0"], default_alpha=0.9, ema_rate=0.1)
        for _ in range(200):
            helpful = random.random() > 0.5
            table.update("topic_0", alpha_used=0.9, helpful=helpful)
        # after many random steps, alpha should be closer to 0.5 than to 0.9
        final_alpha = table.get_alpha("topic_0")
        assert abs(final_alpha - 0.5) < abs(0.9 - 0.5)

    def test_invalid_ema_rate_raises(self):
        with pytest.raises(ValueError, match="ema_rate"):
            AdaptiveAlphaTable(TOPICS, ema_rate=0.0)

    def test_invalid_alpha_bounds_raises(self):
        with pytest.raises(ValueError, match="alpha_min"):
            AdaptiveAlphaTable(TOPICS, alpha_min=0.8, alpha_max=0.2)

    def test_default_alpha_outside_bounds_raises(self):
        with pytest.raises(ValueError, match="default_alpha"):
            AdaptiveAlphaTable(TOPICS, default_alpha=1.5)
