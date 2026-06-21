"""AutoML utilities for CSAI415 D1.

This module intentionally keeps the retriever fitting/evaluation logic outside the
Optuna wrapper. The objective receives builder/evaluator callbacks so the same
code can be reused with the current synthetic corpus and later with the real PDF
retrieval stack.
"""

from __future__ import annotations

import json
import platform
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Protocol

import numpy as np
import optuna
from optuna.samplers import TPESampler


class HybridKNNRetrieverLike(Protocol):
    """Structural type used only for callbacks in this module."""


@dataclass(frozen=True)
class RetrieverConfig:
    """Configuration for a hybrid kNN retriever.

    alpha is interpreted as the lexical/BM25 weight:
        final_score = alpha * lexical_score + (1 - alpha) * dense_score

    The dense side may be a TF-IDF + TruncatedSVD representation during D1.
    """

    k: int
    metric: str
    svd_dim: int
    normalize: bool
    alpha: float


def latency_penalty(
    p95_ms: float,
    *,
    latency_lambda: float = 0.05,
    latency_grace_ms: float = 1000.0,
    latency_scale_ms: float = 1000.0,
) -> float:
    """Soft latency penalty applied only after the grace threshold.

    Examples with defaults:
    - p95_ms=180  -> 0.000
    - p95_ms=1000 -> 0.000
    - p95_ms=2000 -> 0.050
    """

    return latency_lambda * max(
        0.0,
        (p95_ms - latency_grace_ms) / latency_scale_ms,
    )


def make_optuna_objective(
    *,
    build_retriever: Callable[[RetrieverConfig], HybridKNNRetrieverLike],
    evaluate_retriever: Callable[[HybridKNNRetrieverLike], dict[str, float]],
    latency_grace_ms: float = 1000.0,
    latency_scale_ms: float = 1000.0,
    latency_lambda: float = 0.05,
) -> Callable[[optuna.Trial], float]:
    """Create the Optuna objective closure for Track A.

    The objective samples retriever hyperparameters, builds a retriever via the
    supplied callback, evaluates it via the supplied callback, and returns:

        NDCG@5 - lambda * max(0, (p95_ms - grace_ms) / scale_ms)

    No fitting logic is included here by design.
    """

    def objective(trial: optuna.Trial) -> float:
        config = RetrieverConfig(
            k=trial.suggest_categorical("k", [3, 5, 10, 15]),
            metric=trial.suggest_categorical("metric", ["cosine", "euclidean"]),
            svd_dim=trial.suggest_categorical("svd_dim", [16, 32, 64, 96, 128]),
            normalize=trial.suggest_categorical("normalize", [True, False]),
            alpha=trial.suggest_float("alpha", 0.0, 1.0, step=0.02),
        )

        retriever = build_retriever(config)
        metrics = evaluate_retriever(retriever)

        ndcg_at_5 = metrics["ndcg@5"]
        p95_ms = metrics["p95_ms"]
        penalty = latency_penalty(
            p95_ms,
            latency_lambda=latency_lambda,
            latency_grace_ms=latency_grace_ms,
            latency_scale_ms=latency_scale_ms,
        )
        score = ndcg_at_5 - penalty

        trial.set_user_attr("config", asdict(config))
        trial.set_user_attr("ndcg@5", ndcg_at_5)
        trial.set_user_attr("p95_ms", p95_ms)
        trial.set_user_attr("latency_penalty", penalty)
        trial.set_user_attr("penalised_ndcg@5", score)

        if "recall@5" in metrics:
            trial.set_user_attr("recall@5", metrics["recall@5"])

        return score

    return objective


def run_automl_study(
    *,
    build_retriever: Callable[[RetrieverConfig], HybridKNNRetrieverLike],
    evaluate_retriever: Callable[[HybridKNNRetrieverLike], dict[str, float]],
    n_trials: int = 30,
    seed: int = 42,
    study_name: str = "hybrid_knn_automl",
    output_dir: str | Path = "runs/automl",
    latency_grace_ms: float = 1000.0,
    latency_scale_ms: float = 1000.0,
    latency_lambda: float = 0.05,
    save_outputs: bool = True,
) -> optuna.Study:
    """Run a reproducible Optuna study using TPESampler.

    Reproducibility controls:
    - Python and NumPy RNGs are seeded.
    - TPESampler uses the same seed and 10 startup trials.
    - Optuna runs sequentially with n_jobs=1.
    - A JSON run summary and CSV trials table are optionally saved.
    """

    random.seed(seed)
    np.random.seed(seed)

    sampler = TPESampler(seed=seed, n_startup_trials=10)

    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
    )

    objective = make_optuna_objective(
        build_retriever=build_retriever,
        evaluate_retriever=evaluate_retriever,
        latency_grace_ms=latency_grace_ms,
        latency_scale_ms=latency_scale_ms,
        latency_lambda=latency_lambda,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=1,
        show_progress_bar=False,
    )

    if save_outputs:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = {
            "study_name": study_name,
            "seed": seed,
            "n_trials": n_trials,
            "sampler": {
                "name": "TPESampler",
                "seed": seed,
                "n_startup_trials": 10,
            },
            "best_trial": {
                "number": study.best_trial.number,
                "value": study.best_trial.value,
                "params": study.best_trial.params,
                "user_attrs": dict(study.best_trial.user_attrs),
            },
            "environment": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "optuna_version": optuna.__version__,
                "numpy_version": np.__version__,
            },
        }

        with (output_dir / f"{study_name}_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        study.trials_dataframe().to_csv(
            output_dir / f"{study_name}_trials.csv",
            index=False,
        )

    return study


def build_run_card(
    *,
    study: optuna.Study,
    eval_metrics: dict[str, float],
    baseline_metrics: dict[str, float],
    seed: int = 42,
    run_name: str = "hybrid_knn_automl",
) -> dict[str, Any]:
    """Build a YAML-safe run card for the D1 AutoML report."""

    best_trial = study.best_trial
    metric_deltas: dict[str, dict[str, float | None]] = {}

    for metric_name, tuned_value in eval_metrics.items():
        baseline_value = baseline_metrics.get(metric_name)
        if baseline_value is None:
            continue

        absolute_delta = tuned_value - baseline_value
        relative_delta_pct = (
            (absolute_delta / baseline_value) * 100
            if baseline_value != 0
            else None
        )

        metric_deltas[metric_name] = {
            "baseline": baseline_value,
            "tuned": tuned_value,
            "absolute_delta": absolute_delta,
            "relative_delta_pct": relative_delta_pct,
        }

    return {
        "run_name": run_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reproducibility": {
            "seed": seed,
            "optuna_sampler": "TPESampler",
            "n_startup_trials": 10,
            "n_jobs": 1,
            "note": (
                "Use the same seed, fixed data order, sequential Optuna trials, "
                "and pinned package versions for comparable runs."
            ),
        },
        "study": {
            "study_name": study.study_name,
            "direction": study.direction.name,
            "n_trials": len(study.trials),
            "best_trial_number": best_trial.number,
            "best_objective_value": best_trial.value,
            "best_params": dict(best_trial.params),
            "best_user_attrs": dict(best_trial.user_attrs),
        },
        "metrics": {
            "baseline": baseline_metrics,
            "tuned": eval_metrics,
            "delta_over_baseline": metric_deltas,
        },
        "objective": {
            "name": "penalised_ndcg@5",
            "formula": (
                "NDCG@5 - lambda * max(0, "
                "(p95_ms - latency_grace_ms) / latency_scale_ms)"
            ),
            "primary_metric": "ndcg@5",
            "latency_metric": "p95_ms",
        },
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "optuna_version": optuna.__version__,
            "numpy_version": np.__version__,
        },
    }
