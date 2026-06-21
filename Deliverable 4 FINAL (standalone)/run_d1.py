"""CSAI415 D1 — end-to-end orchestration script.

Runs all four stages in order and saves every artefact the D1 deliverable
requires.  Import structure is intentional: every module is imported, never
copy-pasted, so fixes in the library code take effect here automatically.

Stages
------
  1. Corpus build        — synthetic chunks + gold queries + query stream
  2. Baseline evaluation — default RetrieverConfig, no tuning
  3. AutoML study        — Optuna Track A, 50 trials, TPESampler(seed=42)
  4. Online learning     — OnlineTopicClassifier + AdaptiveAlphaTable over
                           the 400-step stream

Outputs  (all written to --output-dir, default runs/d1/)
-------
  run_card.json          — YAML-safe run summary for the D1 report
  automl_trials.csv      — per-trial Optuna dataframe
  per_query_baseline.csv — per-query quality + latency for the baseline
  per_query_best.csv     — same for the AutoML best config
  stream_acc.csv         — (step, accuracy) for the prequential figure
  alpha_history.csv      — full AdaptiveAlphaTable history for figure 5
  summary.txt            — the console summary table, also written to disk

Usage
-----
  python run_d1.py                         # all defaults
  python run_d1.py --trials 50             # override trial count
  python run_d1.py --output-dir my/runs    # custom output dir
  python run_d1.py --seed 7               # different seed
  python run_d1.py --no-automl            # skip Optuna (quick smoke test)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# project imports  (all library code lives here; nothing is copy-pasted)
# ---------------------------------------------------------------------------

from src.automl_utils import RetrieverConfig, build_run_card, run_automl_study
from src.data_utils import Query, build_corpus, build_query_stream
from src.evaluation import evaluate_retriever, evaluate_retriever_per_query
from src.online_learner import AdaptiveAlphaTable, OnlineTopicClassifier
from src.retriever import HybridKNNRetriever


# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

# Default RetrieverConfig used as the baseline before any AutoML tuning.
# Chosen to be a neutral mid-point: cosine similarity, moderate SVD dimension,
# equal BM25/dense weight, L2 normalisation on.
BASELINE_CONFIG = RetrieverConfig(
    k=5,
    metric="cosine",
    svd_dim=32,
    normalize=True,
    alpha=0.5,
)

EVAL_K        = 5     # rank cut-off for all retrieval metrics
EVAL_REPEATS  = 5     # timing repeats per query  (40 queries × 5 = 200 samples)
STREAM_LEN    = 400   # total steps in the query stream
DRIFT_AT      = 200   # step at which the topic distribution narrows
OL_WINDOW     = 50    # prequential accuracy rolling window
OL_DELTA      = 0.002 # ADWIN confidence parameter
OL_COOLDOWN   = 30    # minimum steps between successive ADWIN detections


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _header(text: str) -> None:
    """Print a section header to stdout."""
    bar = "─" * 60
    print(f"\n{bar}")
    print(f"  {text}")
    print(bar)


def _elapsed(t0: float) -> str:
    return f"{time.perf_counter() - t0:.1f}s"


def _save_csv(rows: List[Dict], path: Path) -> None:
    """Write a list-of-dicts to CSV.  Infers fieldnames from the first row."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _retriever_from_config(
    config: RetrieverConfig,
    chunks: list,
) -> HybridKNNRetriever:
    return HybridKNNRetriever(config).fit(chunks)


def _summary_table(
    baseline: Dict[str, float],
    best: Dict[str, float],
    best_config: RetrieverConfig,
    drift_indices: List[int],
    true_drift_step: int,
    n_trials: int,
) -> str:
    """Build the console summary table as a plain string.

    Returns the string so it can be both printed and written to summary.txt.
    """
    lines: List[str] = []
    W = 62  # total table width

    def rule(char="─"):
        return char * W

    def row(label: str, base: str, best_val: str, delta: str = "") -> str:
        return f"  {label:<22}  {base:>10}  {best_val:>10}  {delta:>10}"

    lines += [
        "",
        "╔" + "═" * (W - 2) + "╗",
        "║" + "  CSAI415 D1 — RESULTS SUMMARY".center(W - 2) + "║",
        "╚" + "═" * (W - 2) + "╝",
        "",
        rule(),
        row("Metric", "Baseline", "AutoML best", "Δ"),
        rule(),
    ]

    metric_keys = [
        (f"ndcg@{EVAL_K}",   "NDCG@5"),
        (f"recall@{EVAL_K}", "Recall@5"),
        ("mrr",              "MRR"),
    ]

    for key, label in metric_keys:
        b = baseline.get(key, float("nan"))
        a = best.get(key, float("nan"))
        delta = a - b
        sign  = "+" if delta >= 0 else ""
        lines.append(
            row(label, f"{b:.4f}", f"{a:.4f}", f"{sign}{delta:.4f}")
        )

    lines.append(rule())

    lat_keys = [
        ("p95_ms",  "p95 latency (ms)"),
        ("mean_ms", "mean latency (ms)"),
    ]
    for key, label in lat_keys:
        b = baseline.get(key, float("nan"))
        a = best.get(key, float("nan"))
        delta = a - b
        sign  = "+" if delta >= 0 else ""
        lines.append(
            row(label, f"{b:.1f}", f"{a:.1f}", f"{sign}{delta:.1f}")
        )

    lines += [
        rule(),
        "",
        "  AutoML study",
        f"    Trials run      : {n_trials}",
        f"    Best config     : k={best_config.k}, metric={best_config.metric!r},",
        f"                      svd_dim={best_config.svd_dim},",
        f"                      normalize={best_config.normalize},",
        f"                      alpha={best_config.alpha:.2f}",
        "",
        "  Online learning",
        f"    True drift step : {true_drift_step}",
        f"    ADWIN detections: {len(drift_indices)}",
    ]

    if drift_indices:
        lags = [d - true_drift_step for d in drift_indices]
        lag_str = ", ".join(
            f"step {d} (lag {'+' if l >= 0 else ''}{l})"
            for d, l in zip(drift_indices, lags)
        )
        lines.append(f"    Detection detail: {lag_str}")

    lines += ["", rule(), ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# stage 1 — corpus
# ---------------------------------------------------------------------------

def stage_corpus(seed: int) -> Tuple[list, list, list]:
    """Build chunks, gold queries, and the temporal query stream."""
    t0 = time.perf_counter()
    chunks, queries = build_corpus(n_papers=80, chunks_per_paper=5, seed=seed)
    stream = build_query_stream(
        queries, n_stream=STREAM_LEN, drift_at=DRIFT_AT, seed=seed
    )
    print(f"  Corpus  : {len(chunks)} chunks, {len(queries)} gold queries  [{_elapsed(t0)}]")
    print(f"  Stream  : {len(stream)} steps, drift injected at step {DRIFT_AT}")
    return chunks, queries, stream


# ---------------------------------------------------------------------------
# stage 2 — baseline evaluation
# ---------------------------------------------------------------------------

def stage_baseline(
    chunks: list,
    queries: list,
) -> Tuple[Dict[str, float], List[Dict]]:
    """Fit baseline retriever and evaluate it."""
    t0 = time.perf_counter()
    retriever = _retriever_from_config(BASELINE_CONFIG, chunks)
    metrics   = evaluate_retriever(
        retriever, queries, k=EVAL_K, repeats=EVAL_REPEATS
    )
    per_query = evaluate_retriever_per_query(
        retriever, queries, k=EVAL_K, repeats=EVAL_REPEATS
    )
    print(
        f"  NDCG@{EVAL_K}={metrics[f'ndcg@{EVAL_K}']:.4f}  "
        f"Recall@{EVAL_K}={metrics[f'recall@{EVAL_K}']:.4f}  "
        f"MRR={metrics['mrr']:.4f}  "
        f"p95={metrics['p95_ms']:.1f}ms  [{_elapsed(t0)}]"
    )
    return metrics, per_query


# ---------------------------------------------------------------------------
# stage 3 — AutoML study
# ---------------------------------------------------------------------------

def stage_automl(
    chunks: list,
    queries: list,
    n_trials: int,
    seed: int,
    output_dir: Path,
) -> Tuple[Dict[str, float], List[Dict], RetrieverConfig, object]:
    """Run the Optuna study and evaluate the best config."""

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    t0 = time.perf_counter()

    def build_retriever(config: RetrieverConfig) -> HybridKNNRetriever:
        return _retriever_from_config(config, chunks)

    def evaluate_retriever_cb(retriever: HybridKNNRetriever) -> Dict[str, float]:
        return evaluate_retriever(retriever, queries, k=EVAL_K, repeats=3)

    study = run_automl_study(
        build_retriever=build_retriever,
        evaluate_retriever=evaluate_retriever_cb,
        n_trials=n_trials,
        seed=seed,
        study_name="d1_hybrid_knn",
        output_dir=output_dir,
        save_outputs=True,
    )

    # Re-evaluate best config with full repeats for the final run card.
    best_params = study.best_trial.params
    best_config = RetrieverConfig(
        k=best_params["k"],
        metric=best_params["metric"],
        svd_dim=best_params["svd_dim"],
        normalize=best_params["normalize"],
        alpha=best_params["alpha"],
    )
    best_retriever = _retriever_from_config(best_config, chunks)
    best_metrics   = evaluate_retriever(
        best_retriever, queries, k=EVAL_K, repeats=EVAL_REPEATS
    )
    best_per_query = evaluate_retriever_per_query(
        best_retriever, queries, k=EVAL_K, repeats=EVAL_REPEATS
    )

    print(
        f"  Best trial #{study.best_trial.number}: "
        f"NDCG@{EVAL_K}={best_metrics[f'ndcg@{EVAL_K}']:.4f}  "
        f"Recall@{EVAL_K}={best_metrics[f'recall@{EVAL_K}']:.4f}  "
        f"MRR={best_metrics['mrr']:.4f}  "
        f"p95={best_metrics['p95_ms']:.1f}ms  [{_elapsed(t0)}]"
    )
    print(
        f"  Config  : k={best_config.k}, metric={best_config.metric!r}, "
        f"svd_dim={best_config.svd_dim}, "
        f"normalize={best_config.normalize}, alpha={best_config.alpha:.2f}"
    )

    return best_metrics, best_per_query, best_config, study


# ---------------------------------------------------------------------------
# stage 4 — online learning simulation
# ---------------------------------------------------------------------------

def stage_online_learning(
    stream: list,
    queries: list,
) -> Tuple[List[Tuple[int, float]], List[int], object]:
    """Run the query stream through OnlineTopicClassifier + AdaptiveAlphaTable.

    Returns
    -------
    rolling_acc   : list of (step, accuracy) — for the prequential figure
    drift_indices : list of steps where ADWIN fired
    alpha_table   : fitted AdaptiveAlphaTable (call .alpha_history() for CSV)
    """
    t0 = time.perf_counter()

    topic_labels = sorted({q.topic_id for q in queries})

    clf = OnlineTopicClassifier(
        topic_labels,
        delta=OL_DELTA,
        window_size=OL_WINDOW,
        cooldown=OL_COOLDOWN,
    )
    alpha_table = AdaptiveAlphaTable(
        topic_labels,
        default_alpha=0.5,
        ema_rate=0.1,
    )

    rolling_acc: List[Tuple[int, float]] = []

    for query in stream:
        drift_fired = clf.learn(query.query_text, query.topic_id)
        acc = clf.prequential_accuracy()
        if acc is not None:
            rolling_acc.append((clf.n_seen, acc))

        # Adaptive alpha — proxy helpful signal for D1:
        # treat a query as "helpful" if its topic is in the non-drift half.
        # In production this comes from a real click/feedback signal.
        alpha = alpha_table.get_alpha(query.topic_id)
        helpful = query.topic_id not in [f"topic_{i}" for i in range(4)]
        alpha_table.update(query.topic_id, alpha_used=alpha, helpful=helpful)

    drift_indices = clf.drift_indices
    final_acc = rolling_acc[-1][1] if rolling_acc else float("nan")

    print(
        f"  Steps: {clf.n_seen}  |  "
        f"ADWIN detections: {len(drift_indices)}  |  "
        f"Final acc: {final_acc:.4f}  [{_elapsed(t0)}]"
    )

    if drift_indices:
        for idx in drift_indices:
            lag = idx - DRIFT_AT
            sign = "+" if lag >= 0 else ""
            print(f"    → Detection at step {idx}  (lag {sign}{lag} vs true drift at {DRIFT_AT})")

    return rolling_acc, drift_indices, alpha_table


# ---------------------------------------------------------------------------
# save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    output_dir: Path,
    *,
    baseline_metrics: Dict[str, float],
    best_metrics: Dict[str, float],
    best_config: RetrieverConfig,
    study,
    per_query_baseline: List[Dict],
    per_query_best: List[Dict],
    rolling_acc: List[Tuple[int, float]],
    alpha_table: AdaptiveAlphaTable,
    drift_indices: List[int],
    n_trials: int,
    seed: int,
) -> None:
    """Write all D1 deliverable artefacts to output_dir."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── run card ─────────────────────────────────────────────────────────────
    run_card = build_run_card(
        study=study,
        eval_metrics=best_metrics,
        baseline_metrics=baseline_metrics,
        seed=seed,
        run_name="d1_hybrid_knn",
    )
    run_card["online_learning"] = {
        "true_drift_step": DRIFT_AT,
        "drift_indices": drift_indices,
        "detection_lags": [d - DRIFT_AT for d in drift_indices],
        "n_stream_steps": STREAM_LEN,
        "adwin_delta": OL_DELTA,
        "window_size": OL_WINDOW,
        "cooldown": OL_COOLDOWN,
    }
    with (output_dir / "run_card.json").open("w", encoding="utf-8") as fh:
        json.dump(run_card, fh, indent=2)
    print(f"  run_card.json")

    # ── per-query CSVs ────────────────────────────────────────────────────────
    # Strip list fields (retrieved_ids, relevant_ids) before writing CSV
    # because csv.DictWriter doesn't serialise lists cleanly.
    def _flatten(rows: List[Dict]) -> List[Dict]:
        out = []
        for r in rows:
            flat = {
                k: ("|".join(v) if isinstance(v, list) else v)
                for k, v in r.items()
            }
            out.append(flat)
        return out

    _save_csv(_flatten(per_query_baseline), output_dir / "per_query_baseline.csv")
    print(f"  per_query_baseline.csv")

    _save_csv(_flatten(per_query_best), output_dir / "per_query_best.csv")
    print(f"  per_query_best.csv")

    # ── stream accuracy CSV ───────────────────────────────────────────────────
    acc_rows = [{"step": s, "accuracy": a} for s, a in rolling_acc]
    _save_csv(acc_rows, output_dir / "stream_acc.csv")
    print(f"  stream_acc.csv  ({len(acc_rows)} rows)")

    # ── alpha history CSV ─────────────────────────────────────────────────────
    alpha_rows = [
        {"topic_id": t, "global_step": g, "alpha": a, "helpful": int(h)}
        for t, g, a, h in alpha_table.alpha_history()
    ]
    _save_csv(alpha_rows, output_dir / "alpha_history.csv")
    print(f"  alpha_history.csv  ({len(alpha_rows)} rows)")

    # ── summary table ─────────────────────────────────────────────────────────
    table = _summary_table(
        baseline=baseline_metrics,
        best=best_metrics,
        best_config=best_config,
        drift_indices=drift_indices,
        true_drift_step=DRIFT_AT,
        n_trials=n_trials,
    )
    (output_dir / "summary.txt").write_text(table, encoding="utf-8")
    print(f"  summary.txt")

    return table


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CSAI415 D1 — end-to-end orchestration script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--trials", type=int, default=50,
        help="Number of Optuna trials.  Minimum 30 for stable fANOVA importances.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Global random seed (corpus, stream, Optuna sampler).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("runs/d1"),
        help="Directory for all output artefacts.",
    )
    p.add_argument(
        "--no-automl", action="store_true",
        help="Skip the Optuna study (useful for quick smoke tests).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    print(f"\nCSAI415 D1 — run_d1.py")
    print(f"  seed={args.seed}  trials={args.trials}  output={args.output_dir}")

    # ── stage 1 ───────────────────────────────────────────────────────────────
    _header("Stage 1 / 4 — Corpus build")
    chunks, queries, stream = stage_corpus(seed=args.seed)

    # ── stage 2 ───────────────────────────────────────────────────────────────
    _header("Stage 2 / 4 — Baseline evaluation")
    baseline_metrics, per_query_baseline = stage_baseline(chunks, queries)

    # ── stage 3 ───────────────────────────────────────────────────────────────
    if args.no_automl:
        _header("Stage 3 / 4 — AutoML study  [SKIPPED — --no-automl]")
        # Use baseline config as stand-in so downstream stages still run.
        best_metrics    = baseline_metrics
        per_query_best  = per_query_baseline
        best_config     = BASELINE_CONFIG
        study           = None
    else:
        _header("Stage 3 / 4 — AutoML study  (Optuna, TPESampler)")
        best_metrics, per_query_best, best_config, study = stage_automl(
            chunks, queries,
            n_trials=args.trials,
            seed=args.seed,
            output_dir=args.output_dir,
        )

    # ── stage 4 ───────────────────────────────────────────────────────────────
    _header("Stage 4 / 4 — Online learning simulation")
    rolling_acc, drift_indices, alpha_table = stage_online_learning(stream, queries)

    # ── save outputs ──────────────────────────────────────────────────────────
    _header("Saving outputs")

    if study is None:
        # --no-automl: create a minimal stub so save_outputs doesn't crash.
        import optuna as _optuna
        study = _optuna.create_study()
        study.add_trial(
            _optuna.trial.create_trial(
                params={
                    "k": BASELINE_CONFIG.k,
                    "metric": BASELINE_CONFIG.metric,
                    "svd_dim": BASELINE_CONFIG.svd_dim,
                    "normalize": BASELINE_CONFIG.normalize,
                    "alpha": BASELINE_CONFIG.alpha,
                },
                distributions={
                    "k": _optuna.distributions.CategoricalDistribution([3, 5, 10, 15]),
                    "metric": _optuna.distributions.CategoricalDistribution(["cosine", "euclidean"]),
                    "svd_dim": _optuna.distributions.CategoricalDistribution([16, 32, 64, 96, 128]),
                    "normalize": _optuna.distributions.CategoricalDistribution([True, False]),
                    "alpha": _optuna.distributions.FloatDistribution(0.0, 1.0, step=0.02),
                },
                value=baseline_metrics.get(f"ndcg@{EVAL_K}", 0.0),
            )
        )

    table = save_outputs(
        args.output_dir,
        baseline_metrics=baseline_metrics,
        best_metrics=best_metrics,
        best_config=best_config,
        study=study,
        per_query_baseline=per_query_baseline,
        per_query_best=per_query_best,
        rolling_acc=rolling_acc,
        alpha_table=alpha_table,
        drift_indices=drift_indices,
        n_trials=args.trials,
        seed=args.seed,
    )

    # ── summary table ─────────────────────────────────────────────────────────
    _header("Summary")
    print(table)


if __name__ == "__main__":
    main()
