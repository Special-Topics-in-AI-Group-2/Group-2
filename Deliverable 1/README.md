# CSAI415 D1 — Streaming Learner & AutoML

Deliverable 1 submission for CSAI415.  Contains the full retrieval, AutoML,
evaluation, online learning, and figure pipeline for the D1 report.

## Project structure

```text
run_d1.py                        # one-command orchestration (start here)
requirements.txt                 # pinned dependencies

src/
  data_utils.py                  # Chunk/Query dataclasses, corpus + stream generators
  automl_utils.py                # RetrieverConfig, Optuna objective, run card builder
  retriever.py                   # HybridKNNRetriever — TF-IDF+SVD, WSF, RRF fusion
  online_learner.py              # OnlineTopicClassifier (ADWIN) + AdaptiveAlphaTable
  metrics.py                     # ndcg_at_k, recall_at_k, mean_reciprocal_rank
  evaluation.py                  # evaluate_retriever + per-query breakdown
  figures/
    fig_prequential.py           # Figure 1 — prequential accuracy + ADWIN detections

tests/
  conftest.py
  test_data_utils.py
  test_automl_utils.py
  test_online_learner.py
  test_metrics.py
  test_evaluation.py

reports/
  d1_report.md                   # 2-page D1 report (fill [FILL] placeholders after run)

runs/d1/                         # auto-created by run_d1.py
  run_card.json
  d1_hybrid_knn_trials.csv
  per_query_baseline.csv
  per_query_best.csv
  stream_acc.csv
  alpha_history.csv
  summary.txt
```

## Quickstart

```bash
pip install -r requirements.txt
python run_d1.py                        # all defaults (50 trials)
python run_d1.py --trials 80            # stabler fANOVA importances
python run_d1.py --no-automl            # skip Optuna (smoke test)
pytest -q                               # run all tests
```

## Retriever

TF-IDF → TruncatedSVD (random_state=42) → optional L2 norm → NearestNeighbors.
Two fusion modes: WSF (weighted score fusion, uses alpha) and RRF (reciprocal
rank fusion, pass rrf_k=60).

## AutoML (Track A)

Optuna TPESampler(seed=42, n_startup_trials=10), n_jobs=1.
Search space: k, metric, svd_dim, normalize, alpha.
Objective: NDCG@5 - 0.05 * max(0, (p95_ms - 1000) / 1000).

## Online learning

OnlineTopicClassifier: River MultinomialNB + BagOfWords, ADWIN(delta=0.002),
prequential evaluate-then-train, cooldown=30, model reset on drift.

AdaptiveAlphaTable: per-topic EMA (rate=0.1), retreats to 0.5 on negative
feedback, bounds [0.05, 0.95].

## Reproducibility

| Component        | Control                                      |
|------------------|----------------------------------------------|
| Corpus / stream  | random.seed(seed) in build_corpus/stream     |
| TruncatedSVD     | random_state=42 in HybridKNNRetriever.fit()  |
| Optuna sampler   | TPESampler(seed=seed, n_startup_trials=10)   |
| NumPy RNG        | np.random.seed(seed) in run_automl_study()   |
| Trial execution  | n_jobs=1 (sequential)                        |

## D2 migration

Swap TF-IDF+SVD for bge-small-en + Qdrant:
- fit(): SentenceTransformer.encode() + Qdrant upsert
- _encode_query(): model.encode()
- search() dense lookup: client.search()
All fusion, metrics, evaluation, and Optuna wrappers are unchanged.

## Team

| Member  | Owns                                        |
|---------|---------------------------------------------|
| [Name]  | data_utils.py, retriever.py                 |
| [Name]  | automl_utils.py, run_d1.py                  |
| [Name]  | online_learner.py, figures/                 |
| [Name]  | metrics.py, evaluation.py, report           |
