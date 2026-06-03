"""Prequential accuracy figure for CSAI415 D1 report.

Usage
-----
    from src.data_utils import build_corpus, build_query_stream
    from src.online_learner import OnlineTopicClassifier
    from src.figures.fig_prequential import fig_prequential

    # --- run the stream ---
    chunks, queries = build_corpus(seed=42)
    stream = build_query_stream(queries, n_stream=400, drift_at=200, seed=42)

    topic_labels = sorted({q.topic_id for q in queries})
    clf = OnlineTopicClassifier(topic_labels, delta=0.002, window_size=50, cooldown=30)

    rolling_acc = []
    for query in stream:
        clf.learn(query.query_text, query.topic_id)
        acc = clf.prequential_accuracy()
        if acc is not None:
            rolling_acc.append((clf.n_seen, acc))

    # --- draw ---
    fig = fig_prequential(
        rolling_acc=rolling_acc,
        drift_indices=clf.drift_indices,
        true_drift_step=200,
        window_size=50,
    )
    fig.savefig("reports/fig1_prequential.pdf", bbox_inches="tight")
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# colour palette  (all defined here so tweaks stay in one place)
# ---------------------------------------------------------------------------

_C_ACCURACY    = "#2563EB"   # blue  — accuracy line
_C_TRUE_DRIFT  = "#6B7280"   # gray  — true injection point (dashed)
_C_ADWIN       = "#F97316"   # orange — ADWIN detection lines (dotted)
_C_PRE_DRIFT   = "#DBEAFE"   # light blue tint — pre-drift shaded region
_C_POST_DRIFT  = "#FEF3C7"   # light amber tint — post-drift shaded region
_C_GRID        = "#E5E7EB"   # light gray — grid lines


def fig_prequential(
    rolling_acc: List[Tuple[int, float]],
    drift_indices: List[int],
    true_drift_step: int = 200,
    window_size: int = 50,
    *,
    figsize: Tuple[float, float] = (10, 4),
    title: str = "Prequential Accuracy with ADWIN Drift Detection",
    show_shading: bool = True,
    y_min: Optional[float] = None,
    y_max: Optional[float] = None,
) -> Figure:
    """Draw the prequential accuracy curve for the D1 report.

    The figure shows:
      - Rolling accuracy line (blue) computed with the evaluate-then-train
        prequential protocol.  Points before the warm-up window are excluded
        (caller should only pass steps where prequential_accuracy() returned
        a non-None value).
      - True drift injection point (dashed gray vertical line).  This is the
        step at which build_query_stream() narrows the topic distribution.
      - ADWIN detection points (dotted orange vertical lines).  One line per
        entry in drift_indices.  The gap between the first orange line and
        the gray line is the detection lag.
      - Pre/post drift shading (optional).  Light blue before the true drift
        step, light amber after.  Helps the reader parse the three phases:
        stable → drift → recovery.

    Parameters
    ----------
    rolling_acc : list of (step, accuracy) tuples
        Collected by calling clf.prequential_accuracy() after each
        clf.learn() call and keeping non-None values.
        step is clf.n_seen at the time of collection.
    drift_indices : list[int]
        clf.drift_indices after the stream completes.  May be empty if
        ADWIN found no drift (that itself is a result worth showing).
    true_drift_step : int
        The drift_at argument passed to build_query_stream().  Default 200.
    window_size : int
        The window_size used for OnlineTopicClassifier.  Shown in the
        y-axis label so the reader knows the smoothing scale.
    figsize : (width, height) in inches.  Default (10, 4).
    title : str  Figure title.
    show_shading : bool
        When True (default), shade pre/post drift regions.  Set False for
        a cleaner look in two-column report layouts.
    y_min, y_max : float | None
        Manual y-axis limits.  When None the limits are set automatically
        with a small margin below the minimum observed accuracy.

    Returns
    -------
    matplotlib.figure.Figure
        The caller is responsible for saving or showing the figure.

    Notes
    -----
    Detection lag (first ADWIN fire − true drift step) is annotated on the
    figure when at least one detection exists.  If ADWIN fires before the
    true drift step it means the random stream had an accidental accuracy
    dip — worth noting in the report.
    """
    if not rolling_acc:
        raise ValueError("rolling_acc is empty — run the stream first.")

    steps, accs = zip(*rolling_acc)
    steps = list(steps)
    accs  = list(accs)

    x_min, x_max = steps[0], steps[-1]

    # ── figure / axes ─────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)

    # ── 1. pre / post drift shading ───────────────────────────────────────────
    if show_shading:
        # Pre-drift region: from start of accuracy data to true drift step.
        ax.axvspan(
            x_min, true_drift_step,
            color=_C_PRE_DRIFT, alpha=0.6, zorder=0,
            label="Pre-drift (stable distribution)",
        )
        # Post-drift region: from true drift step to end of stream.
        ax.axvspan(
            true_drift_step, x_max,
            color=_C_POST_DRIFT, alpha=0.6, zorder=0,
            label="Post-drift (narrowed distribution)",
        )

    # ── 2. accuracy line ──────────────────────────────────────────────────────
    ax.plot(
        steps, accs,
        color=_C_ACCURACY, linewidth=1.8, zorder=3,
        label=f"Rolling accuracy (window={window_size})",
    )

    # ── 3. true drift injection line ─────────────────────────────────────────
    ax.axvline(
        true_drift_step,
        color=_C_TRUE_DRIFT, linewidth=1.5,
        linestyle="--", zorder=4,
        label=f"True drift injection (step {true_drift_step})",
    )

    # ── 4. ADWIN detection lines ──────────────────────────────────────────────
    # Draw all detection lines first; only add one legend entry.
    for i, det_step in enumerate(drift_indices):
        ax.axvline(
            det_step,
            color=_C_ADWIN, linewidth=1.4,
            linestyle=":", zorder=5,
            label="ADWIN detection" if i == 0 else None,
        )

    # ── 5. detection lag annotation ───────────────────────────────────────────
    if drift_indices:
        first_detection = drift_indices[0]
        lag = first_detection - true_drift_step

        # Choose annotation side so it never overlaps the drift lines.
        text_x     = first_detection + (x_max - x_min) * 0.015
        text_y     = min(accs) + (max(accs) - min(accs)) * 0.08
        ha         = "left"

        lag_sign = "+" if lag >= 0 else ""
        ax.annotate(
            f"Detection lag: {lag_sign}{lag} steps",
            xy=(first_detection, text_y),
            xytext=(text_x, text_y),
            fontsize=8.5,
            color=_C_ADWIN,
            ha=ha, va="bottom",
            arrowprops=dict(
                arrowstyle="->",
                color=_C_ADWIN,
                lw=1.0,
            ),
        )

    # ── 6. axes labels and limits ─────────────────────────────────────────────
    ax.set_xlabel("Stream step", fontsize=11)
    ax.set_ylabel(f"Prequential accuracy\n(window = {window_size})", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)

    # x-axis: full stream range with a small right margin for the annotation.
    ax.set_xlim(x_min, x_max + (x_max - x_min) * 0.02)

    # y-axis: leave a small gap below the minimum so the line isn't clipped.
    obs_min = min(accs)
    obs_max = max(accs)
    margin  = (obs_max - obs_min) * 0.12 if obs_max > obs_min else 0.05

    ax.set_ylim(
        y_min if y_min is not None else max(0.0, obs_min - margin),
        y_max if y_max is not None else min(1.0, obs_max + margin),
    )

    # ── 7. grid ───────────────────────────────────────────────────────────────
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=_C_GRID, linewidth=0.8, zorder=0)
    ax.grid(axis="x", color=_C_GRID, linewidth=0.5, linestyle=":", zorder=0)

    # ── 8. legend ─────────────────────────────────────────────────────────────
    # Build a clean legend even when drift_indices is empty.
    handles, labels = ax.get_legend_handles_labels()

    if not drift_indices:
        # Add a greyed-out placeholder so the legend slot is not empty.
        handles.append(
            Line2D([0], [0], color=_C_ADWIN, linestyle=":", linewidth=1.4)
        )
        labels.append("ADWIN detection (none fired)")

    ax.legend(
        handles, labels,
        loc="lower left",
        fontsize=9,
        framealpha=0.92,
        edgecolor=_C_GRID,
    )

    # ── 9. spine clean-up ─────────────────────────────────────────────────────
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color(_C_GRID)
    ax.spines["bottom"].set_color(_C_GRID)

    fig.tight_layout()
    return fig
