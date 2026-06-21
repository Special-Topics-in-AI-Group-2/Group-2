import pytest

optuna = pytest.importorskip("optuna")

from src.automl_utils import (
    RetrieverConfig,
    build_run_card,
    latency_penalty,
    make_optuna_objective,
    run_automl_study,
)


class DummyRetriever:
    def __init__(self, config: RetrieverConfig):
        self.config = config


def test_latency_penalty_has_grace_period():
    assert latency_penalty(180.0) == 0.0
    assert latency_penalty(1000.0) == 0.0
    assert latency_penalty(2000.0) == 0.05


def test_objective_returns_penalised_ndcg_without_latency_penalty():
    def build_retriever(config):
        return DummyRetriever(config)

    def evaluate_retriever(_retriever):
        return {
            "ndcg@5": 0.41,
            "recall@5": 0.60,
            "p95_ms": 180.0,
        }

    objective = make_optuna_objective(
        build_retriever=build_retriever,
        evaluate_retriever=evaluate_retriever,
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    assert study.best_value == 0.41
    assert study.best_trial.user_attrs["latency_penalty"] == 0.0


def test_run_automl_study_is_callable_without_saving_outputs():
    def build_retriever(config):
        return DummyRetriever(config)

    def evaluate_retriever(retriever):
        # Deterministic fake score that prefers alpha near 0.5.
        alpha = retriever.config.alpha
        return {
            "ndcg@5": 0.5 - abs(alpha - 0.5) * 0.1,
            "recall@5": 0.7,
            "p95_ms": 180.0,
        }

    study = run_automl_study(
        build_retriever=build_retriever,
        evaluate_retriever=evaluate_retriever,
        n_trials=3,
        seed=42,
        save_outputs=False,
    )

    assert len(study.trials) == 3
    assert "alpha" in study.best_trial.params


def test_build_run_card_contains_best_params_and_delta():
    def objective(trial):
        trial.suggest_int("k", 3, 5)
        return 0.4

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)

    run_card = build_run_card(
        study=study,
        eval_metrics={"ndcg@5": 0.41, "recall@5": 0.62, "p95_ms": 180.0},
        baseline_metrics={"ndcg@5": 0.35, "recall@5": 0.55, "p95_ms": 120.0},
        seed=42,
    )

    assert run_card["reproducibility"]["seed"] == 42
    assert "best_params" in run_card["study"]
    assert run_card["metrics"]["delta_over_baseline"]["ndcg@5"]["absolute_delta"] == 0.06
