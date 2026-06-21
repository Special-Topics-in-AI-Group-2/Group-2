"""Online learning components for CSAI415 D1.

Two classes are provided:

OnlineTopicClassifier
    Incremental query→topic classifier built on River's MultinomialNB with
    BagOfWords features.  Uses ADWIN for drift detection.  Implements the
    evaluate-then-train prequential protocol: predict() is always called
    before learn() so accuracy estimates are unbiased.

AdaptiveAlphaTable
    Per-topic EMA tracker for the hybrid fusion weight alpha (BM25 vs dense).
    Updated from binary helpful/not-helpful feedback.  On positive feedback
    alpha is pulled toward the value that produced the result; on negative
    feedback it retreats toward 0.5 (maximum-uncertainty / equal-weight point).

Typical usage
-------------
    from src.data_utils import build_corpus, build_query_stream
    from src.online_learner import OnlineTopicClassifier, AdaptiveAlphaTable

    chunks, queries = build_corpus()
    stream = build_query_stream(queries, n_stream=400, drift_at=200)

    topic_labels = list({q.topic_id for q in queries})
    clf = OnlineTopicClassifier(topic_labels)
    alpha_table = AdaptiveAlphaTable(topic_labels)

    rolling_acc, drift_steps = [], []

    for query in stream:
        drift_fired = clf.learn(query.query_text, query.topic_id)
        acc = clf.prequential_accuracy()
        if acc is not None:
            rolling_acc.append((clf.n_seen, acc))
        if drift_fired:
            drift_steps.append(clf.n_seen)
"""

from __future__ import annotations

import collections
from typing import Optional

from river import drift, feature_extraction, naive_bayes


# ---------------------------------------------------------------------------
# OnlineTopicClassifier
# ---------------------------------------------------------------------------


class OnlineTopicClassifier:
    """Online query→topic classifier with ADWIN drift detection.

    Implements the prequential (interleaved test-then-train) protocol:
    each example is evaluated *before* the model is updated, so the
    rolling accuracy window is an unbiased estimate of true online error.

    Parameters
    ----------
    topic_labels : list[str]
        All possible topic IDs (e.g. ["topic_0", ..., "topic_7"]).
        The first label is used as a cold-start fallback in predict().
    delta : float
        ADWIN confidence parameter.  Controls sensitivity to drift.
        Lower = fewer false alarms but slower detection.
        Default 0.002 (Bifet & Gavaldà 2007 recommendation).
    window_size : int
        Number of recent steps used for rolling prequential accuracy.
        Default 50.  prequential_accuracy() returns None until this many
        steps have been seen, so cold-start noise is excluded from plots.
    cooldown : int
        Minimum number of steps between successive drift detections.
        Prevents double-detection of a single sharp drift event (e.g. the
        198/203 artifact where ADWIN fires twice in quick succession).
        Default 30.
    """

    def __init__(
        self,
        topic_labels: list[str],
        delta: float = 0.002,
        window_size: int = 50,
        cooldown: int = 30,
    ) -> None:
        if not topic_labels:
            raise ValueError("topic_labels must be non-empty.")

        self._topic_labels = list(topic_labels)
        self._window_size = window_size
        self._cooldown = cooldown

        # River pipeline: raw text → bag-of-words dict → MultinomialNB
        self._bow = feature_extraction.BagOfWords()
        self._model = naive_bayes.MultinomialNB()

        # ADWIN detector receives 1.0 (correct) or 0.0 (incorrect) each step.
        # It tracks the mean of this binary stream and fires when it detects
        # a statistically significant change in the accuracy rate.
        self._adwin = drift.ADWIN(delta=delta)

        # Prequential state
        self._window: collections.deque[int] = collections.deque(maxlen=window_size)
        self._drift_indices: list[int] = []
        self._n_seen: int = 0
        self._last_drift_step: int = -cooldown   # allows detection at step 0

    # ------------------------------------------------------------------
    # predict
    # ------------------------------------------------------------------

    def predict(self, text: str) -> str:
        """Predict topic label for a query string.

        Uses the first label as a cold-start fallback before any training
        has occurred (n_seen == 0).

        Parameters
        ----------
        text : str
            Raw query text.

        Returns
        -------
        str
            Predicted topic_id.
        """
        if self._n_seen == 0:
            return self._topic_labels[0]

        features = self._bow.transform_one(text)
        return self._model.predict_one(features)

    # ------------------------------------------------------------------
    # learn
    # ------------------------------------------------------------------

    def learn(self, text: str, true_topic: str) -> bool:
        """Evaluate prediction, train on example, update drift detector.

        Mandatory evaluate-then-train order:
          1. Predict with the current (pre-update) model state.
          2. Compute correct (1) / incorrect (0).
          3. Update BagOfWords vocabulary and MultinomialNB class counts.
          4. Feed the binary correctness signal to ADWIN.
          5. Append to the rolling prequential accuracy window.
             Record step index if drift is detected (subject to cooldown).

        When drift is detected the model is reset (class-count priors
        discarded) while the BagOfWords vocabulary is retained.  This
        lets the model quickly relearn class distributions on the new
        topic mix without losing the token→feature mapping.

        Parameters
        ----------
        text : str
            Raw query text (same string as passed to predict).
        true_topic : str
            Ground-truth topic label for this query.

        Returns
        -------
        bool
            True if ADWIN detected drift on this step (after cooldown
            suppression), False otherwise.
        """
        # ── 1. Predict BEFORE updating the model ─────────────────────────────
        prediction = self.predict(text)

        # ── 2. Correctness signal ─────────────────────────────────────────────
        correct = int(prediction == true_topic)   # 1 = correct, 0 = incorrect

        # ── 3. Update model ───────────────────────────────────────────────────
        # BagOfWords.learn_one updates the vocabulary; transform_one then yields
        # features that include new tokens from this example.  We call them on
        # separate lines (not chained) because River >= 0.22 changed learn_one
        # to return None instead of self — chaining would crash on newer River.
        self._bow.learn_one(text)
        features = self._bow.transform_one(text)
        self._model.learn_one(features, true_topic)

        # ── 4. Feed binary signal to ADWIN ────────────────────────────────────
        # ADWIN tracks the mean of the correctness stream.  A drop in accuracy
        # (e.g. caused by a topic distribution shift) appears as a change in
        # this mean and triggers drift_detected.
        self._adwin.update(correct)

        # Suppress detections that fall within the cooldown window of the
        # most recent detection.  This collapses double-detections of a
        # single sharp drift event into one signal.
        steps_since_last = self._n_seen - self._last_drift_step
        drift_detected: bool = (
            self._adwin.drift_detected
            and steps_since_last > self._cooldown
        )

        # ── 5. Update prequential window and drift index list ─────────────────
        self._window.append(correct)

        if drift_detected:
            self._drift_indices.append(self._n_seen)
            self._last_drift_step = self._n_seen
            self._reset_model()

        self._n_seen += 1

        return drift_detected

    # ------------------------------------------------------------------
    # prequential_accuracy
    # ------------------------------------------------------------------

    def prequential_accuracy(self) -> Optional[float]:
        """Rolling accuracy over the last window_size steps.

        Returns None during cold-start (fewer steps seen than window_size)
        so callers can skip plotting rather than display misleading zeros.

        Example
        -------
        >>> acc = clf.prequential_accuracy()
        >>> if acc is not None:
        ...     rolling_acc.append((clf.n_seen, acc))
        """
        if self._n_seen < self._window_size:
            return None

        return sum(self._window) / len(self._window)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------

    @property
    def n_seen(self) -> int:
        """Total number of learn() calls so far."""
        return self._n_seen

    @property
    def drift_indices(self) -> list[int]:
        """Step indices where ADWIN fired (after cooldown suppression).

        Returns a copy so the internal list cannot be mutated externally.
        """
        return list(self._drift_indices)

    # ------------------------------------------------------------------
    # private helpers
    # ------------------------------------------------------------------

    def _reset_model(self) -> None:
        """Discard stale class-count priors; retain vocabulary.

        Only MultinomialNB is reset.  BagOfWords vocabulary is kept because
        the token→feature mapping is still valid after a topic distribution
        shift — only the per-class counts are stale.
        """
        self._model = naive_bayes.MultinomialNB()

    def __repr__(self) -> str:
        return (
            f"OnlineTopicClassifier("
            f"n_topics={len(self._topic_labels)}, "
            f"n_seen={self._n_seen}, "
            f"drift_indices={self._drift_indices})"
        )


# ---------------------------------------------------------------------------
# AdaptiveAlphaTable
# ---------------------------------------------------------------------------


class AdaptiveAlphaTable:
    """Per-topic EMA tracker for the hybrid fusion weight alpha.

    Each topic maintains its own alpha value representing the BM25 weight
    in WSF fusion:

        fused_score = alpha * norm(bm25) + (1 - alpha) * norm(dense)

    On positive feedback (helpful=True) alpha is pulled toward the value
    that produced the result — reinforcing what worked.

    On negative feedback (helpful=False) alpha retreats toward 0.5, the
    maximum-uncertainty point where neither modality is preferred.  This
    is the epistemically honest move: a single failure tells us the current
    alpha underperformed but not *which direction* to correct, so we
    retreat to the least committed position rather than overreact.

    Note: 0.5 assumes equal prior capability of both modalities.  A more
    sophisticated version would retreat toward the corpus-wide best alpha
    from the AutoML study rather than a hardcoded value.

    Parameters
    ----------
    topic_labels : list[str]
        All known topic IDs.  Each gets an independent alpha initialised
        to default_alpha.
    default_alpha : float
        Starting alpha for all topics.  0.5 is a neutral prior; pass the
        AutoML best alpha if available.
    ema_rate : float
        EMA learning rate in (0, 1).  Larger = faster adaptation but more
        noise.  0.1 gives effective memory of ~10 steps (1 / ema_rate),
        which adapts within ~20–30 steps post-drift on a 400-step stream.
    alpha_min : float
        Lower bound on alpha.  Prevents collapsing to pure dense retrieval.
    alpha_max : float
        Upper bound on alpha.  Prevents collapsing to pure BM25.
    """

    _UNCERTAIN: float = 0.5   # retreat target on negative feedback

    def __init__(
        self,
        topic_labels: list[str],
        default_alpha: float = 0.5,
        ema_rate: float = 0.1,
        alpha_min: float = 0.05,
        alpha_max: float = 0.95,
    ) -> None:
        if not 0.0 < ema_rate < 1.0:
            raise ValueError(f"ema_rate must be in (0, 1), got {ema_rate}")
        if not alpha_min < alpha_max:
            raise ValueError("alpha_min must be less than alpha_max")
        if not alpha_min <= default_alpha <= alpha_max:
            raise ValueError("default_alpha must be within [alpha_min, alpha_max]")
        if not topic_labels:
            raise ValueError("topic_labels must be non-empty.")

        self._ema_rate = ema_rate
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._default_alpha = default_alpha

        # topic_id → current alpha value
        self._alphas: dict[str, float] = {
            t: default_alpha for t in topic_labels
        }

        # topic_id → number of feedback updates received
        self._update_counts: dict[str, int] = {
            t: 0 for t in topic_labels
        }

        # full history for plotting: (topic_id, global_step, alpha, helpful)
        self._history: list[tuple[str, int, float, bool]] = []
        self._n_updates: int = 0

    # ------------------------------------------------------------------
    # read
    # ------------------------------------------------------------------

    def get_alpha(self, topic_id: str) -> float:
        """Return the current alpha for a topic.

        Falls back to default_alpha for unseen topics so the retriever
        never crashes on a topic introduced after initialisation.

        Parameters
        ----------
        topic_id : str
            Topic identifier.

        Returns
        -------
        float
            Current alpha value in [alpha_min, alpha_max].
        """
        return self._alphas.get(topic_id, self._default_alpha)

    # ------------------------------------------------------------------
    # update
    # ------------------------------------------------------------------

    def update(
        self,
        topic_id: str,
        alpha_used: float,
        helpful: bool,
    ) -> float:
        """Apply one EMA step and return the new alpha.

        EMA update rule
        ---------------
            target = alpha_used   if helpful else 0.5
            new    = (1 - r) * old + r * target

        where r = ema_rate.  Result is clipped to [alpha_min, alpha_max].

        Parameters
        ----------
        topic_id : str
            The topic the query was routed to (from OnlineTopicClassifier
            or ground truth during evaluation).
        alpha_used : float
            The alpha value that produced this retrieval result.
            Used as the EMA target on positive feedback.
        helpful : bool
            True  → pull alpha toward alpha_used (reinforce what worked).
            False → pull alpha toward 0.5 (retreat to maximum uncertainty).

        Returns
        -------
        float
            Updated alpha for this topic, clipped to [alpha_min, alpha_max].
        """
        old_alpha = self.get_alpha(topic_id)
        target = alpha_used if helpful else self._UNCERTAIN

        # EMA step: exponential moving average toward target
        new_alpha = (1.0 - self._ema_rate) * old_alpha + self._ema_rate * target

        # clip to valid range — prevents collapsing to a single modality
        new_alpha = max(self._alpha_min, min(self._alpha_max, new_alpha))

        # persist
        self._alphas[topic_id] = new_alpha
        self._update_counts[topic_id] = self._update_counts.get(topic_id, 0) + 1
        self._history.append((topic_id, self._n_updates, new_alpha, helpful))
        self._n_updates += 1

        return new_alpha

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, dict[str, float | int]]:
        """Snapshot of all topic alphas and update counts.

        Useful for the run card and D1 report table.

        Returns
        -------
        dict
            { topic_id: { "alpha": float, "n_updates": int } }
        """
        return {
            topic: {
                "alpha": round(self._alphas[topic], 4),
                "n_updates": self._update_counts[topic],
            }
            for topic in self._alphas
        }

    def alpha_history(
        self,
        topic_id: str | None = None,
    ) -> list[tuple[str, int, float, bool]]:
        """Full update history, optionally filtered by topic.

        Each entry is (topic_id, global_step, alpha_after_update, helpful).
        Useful for plotting per-topic alpha trajectories over the stream.
        Drift-affected topics (0–3) should show visible shifts post-step-200;
        unaffected topics (4–7) should remain flat — a clean visual story.

        Parameters
        ----------
        topic_id : str | None
            If provided, returns only entries for that topic.
            If None, returns the full cross-topic history.
        """
        if topic_id is None:
            return list(self._history)
        return [row for row in self._history if row[0] == topic_id]

    @property
    def n_updates(self) -> int:
        """Total number of feedback updates received across all topics."""
        return self._n_updates

    def __repr__(self) -> str:
        return (
            f"AdaptiveAlphaTable("
            f"n_topics={len(self._alphas)}, "
            f"n_updates={self._n_updates}, "
            f"ema_rate={self._ema_rate})"
        )
