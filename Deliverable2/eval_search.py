"""eval_search.py — CSAI415 D2 /search endpoint evaluator.

Fires queries at the FastAPI /search endpoint, measures Recall@5 and
latency per query, and prints a clean summary table.

Ground-truth modes (FIX #7)
----------------------------
Two modes are supported:

  seed mode (default)
      Uses hardcoded chunk IDs matching seed.py (p001_c0, p001_c1, etc.).
      Run this immediately after `python seed.py` to verify the stack.

  mongo mode  --mongo-gt
      Discovers real chunk IDs dynamically from MongoDB by matching each
      query's topic keywords against the stored 'title' field.
      Use this after running the real ingest pipeline (python ingest.py).
      Requires: pip install pymongo

Usage
-----
    python eval_search.py                           # seed mode (default)
    python eval_search.py --mongo-gt                # real ingest mode
    python eval_search.py --mongo-uri mongodb://localhost:27017 --mongo-gt
    python eval_search.py --top-k 5 --repeats 3
    python eval_search.py --no-color                # plain text (for CI)

Requirements
------------
    pip install requests pymongo
    (FastAPI server must be running: uvicorn api:app --reload)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("requests not found. Run: pip install requests")


# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------

class _C:
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
    GREEN  = "\033[32m"
    YELLOW = "\033[33m"
    RED    = "\033[31m"
    CYAN   = "\033[36m"
    DIM    = "\033[2m"

def _colorize(text: str, *codes: str, use_color: bool = True) -> str:
    if not use_color:
        return text
    return "".join(codes) + text + _C.RESET


# ---------------------------------------------------------------------------
# Query definitions
# ---------------------------------------------------------------------------

@dataclass
class EvalQuery:
    """One evaluation query with its expected relevant chunk IDs."""
    query_text:         str
    relevant_chunk_ids: list[str]   # ground-truth set — populated at runtime in mongo mode
    label:              str = ""
    topic:              str = ""
    # Keywords used to match papers in MongoDB (mongo-gt mode)
    title_keywords:     list[str] = None

    def __post_init__(self):
        if self.title_keywords is None:
            self.title_keywords = []


# Seed-mode ground truth: IDs match exactly what seed.py inserts.
# Each paper has 2 chunks: p00X_c0 and p00X_c1.
QUERIES_SEED: list[EvalQuery] = [
    EvalQuery(
        query_text="transformer architecture attention mechanism",
        relevant_chunk_ids=["p001_c0", "p001_c1"],
        label="Transformer attention", topic="Transformers",
        title_keywords=["attention", "transformer"],
    ),
    EvalQuery(
        query_text="self-attention network dispensing with recurrence",
        relevant_chunk_ids=["p001_c0"],
        label="No recurrence", topic="Transformers",
        title_keywords=["attention", "transformer"],
    ),
    EvalQuery(
        query_text="multi-head attention representation subspaces",
        relevant_chunk_ids=["p001_c1"],
        label="Multi-head attention", topic="Transformers",
        title_keywords=["attention", "transformer"],
    ),
    EvalQuery(
        query_text="BERT bidirectional pre-training language model",
        relevant_chunk_ids=["p002_c0", "p002_c1"],
        label="BERT pre-training", topic="BERT",
        title_keywords=["bert", "bidirectional"],
    ),
    EvalQuery(
        query_text="pre-train deep representations from unlabelled text",
        relevant_chunk_ids=["p002_c0"],
        label="Unlabelled text", topic="BERT",
        title_keywords=["bert", "bidirectional"],
    ),
    EvalQuery(
        query_text="fine-tuning NLP tasks state-of-the-art results",
        relevant_chunk_ids=["p002_c1"],
        label="Fine-tuning BERT", topic="BERT",
        title_keywords=["bert", "bidirectional"],
    ),
    EvalQuery(
        query_text="retrieval augmented generation knowledge intensive tasks",
        relevant_chunk_ids=["p003_c0", "p003_c1"],
        label="RAG overview", topic="RAG",
        title_keywords=["retrieval", "augmented", "generation"],
    ),
    EvalQuery(
        query_text="dense retriever fetch documents seq2seq generation",
        relevant_chunk_ids=["p003_c0"],
        label="Dense retriever", topic="RAG",
        title_keywords=["retrieval", "augmented", "generation"],
    ),
    EvalQuery(
        query_text="open domain question answering factual grounding",
        relevant_chunk_ids=["p003_c1"],
        label="Open-domain QA", topic="RAG",
        title_keywords=["retrieval", "augmented", "generation"],
    ),
    EvalQuery(
        query_text="NLP deep learning pre-trained models transformers BERT",
        relevant_chunk_ids=["p001_c0", "p001_c1", "p002_c0", "p002_c1"],
        label="NLP broad query", topic="Cross-topic",
        title_keywords=["attention", "bert"],
    ),
]


# ---------------------------------------------------------------------------
# FIX #7 — Dynamic ground-truth discovery from MongoDB
# ---------------------------------------------------------------------------

def _discover_ground_truth(
    queries: list[EvalQuery],
    mongo_uri: str,
    mongo_db: str = "csai415",
    mongo_collection: str = "chunks",
) -> list[EvalQuery]:
    """Replace hardcoded seed chunk IDs with real IDs from MongoDB.

    For each query, finds all chunks whose paper title contains at least one
    of the query's title_keywords (case-insensitive), and uses those as the
    ground-truth relevant set.

    This is a lightweight proxy for a proper gold set — good enough for D2
    smoke-testing after real ingestion. D3 should replace this with a curated
    gold Q/A set.
    """
    try:
        from pymongo import MongoClient
    except ImportError:
        sys.exit("pymongo not found. Run: pip install pymongo")

    print(f"[eval] Discovering ground truth from MongoDB ({mongo_uri}/{mongo_db}.{mongo_collection}) ...")

    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5_000)
    try:
        client.admin.command("ping")
    except Exception as exc:
        sys.exit(f"[eval] Cannot connect to MongoDB: {exc}")

    col = client[mongo_db][mongo_collection]

    updated: list[EvalQuery] = []
    for q in queries:
        if not q.title_keywords:
            updated.append(q)
            continue

        # Build a case-insensitive OR regex across all title keywords
        import re
        pattern = "|".join(re.escape(kw) for kw in q.title_keywords)
        matching_chunks = list(col.find(
            {"title": {"$regex": pattern, "$options": "i"}},
            {"chunk_id": 1, "_id": 0},
        ))
        real_ids = [c["chunk_id"] for c in matching_chunks if c.get("chunk_id")]

        if not real_ids:
            print(f"  [warn] No chunks found for query '{q.label}' "
                  f"(keywords: {q.title_keywords}) — keeping seed IDs")
            updated.append(q)
        else:
            updated.append(EvalQuery(
                query_text=q.query_text,
                relevant_chunk_ids=real_ids,
                label=q.label,
                topic=q.topic,
                title_keywords=q.title_keywords,
            ))
            print(f"  {q.label:<25} → {len(real_ids)} relevant chunk(s) discovered")

    client.close()
    return updated


# ---------------------------------------------------------------------------
# Per-query result
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query:         EvalQuery
    retrieved_ids: list[str]
    recall_at_k:   float
    latency_ms:    float
    status_code:   int
    error:         Optional[str] = None


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & set(relevant))
    return hits / len(relevant)


# ---------------------------------------------------------------------------
# API calls
# ---------------------------------------------------------------------------

def check_health(base_url: str, timeout: float) -> None:
    try:
        r = requests.get(f"{base_url}/health", timeout=timeout)
        r.raise_for_status()
    except requests.ConnectionError:
        sys.exit(
            f"\n[ERROR] Cannot connect to {base_url}\n"
            "  Make sure the API is running:  uvicorn api:app --reload\n"
        )
    except requests.HTTPError as exc:
        sys.exit(f"\n[ERROR] /health returned {exc.response.status_code}: {exc}")


def run_query(
    base_url: str,
    query: EvalQuery,
    top_k: int,
    bm25_weight: float,
    dense_weight: float,
    timeout: float,
    repeats: int,
) -> QueryResult:
    params = {
        "query":        query.query_text,
        "top_k":        top_k,
        "bm25_weight":  bm25_weight,
        "dense_weight": dense_weight,
    }

    latencies: list[float] = []
    last_response = None
    last_status   = 0
    error         = None

    for _ in range(repeats):
        t0 = time.perf_counter()
        try:
            resp = requests.get(f"{base_url}/search", params=params, timeout=timeout)
            latencies.append((time.perf_counter() - t0) * 1_000)
            last_response = resp
            last_status   = resp.status_code
        except requests.RequestException as exc:
            return QueryResult(
                query=query, retrieved_ids=[], recall_at_k=0.0,
                latency_ms=0.0, status_code=0, error=str(exc),
            )

    latency_ms = statistics.median(latencies)
    retrieved_ids: list[str] = []

    if last_response is not None and last_status == 200:
        try:
            data = last_response.json()
            retrieved_ids = [r["chunk_id"] for r in data.get("results", [])]
        except (json.JSONDecodeError, KeyError) as exc:
            error = f"Bad response JSON: {exc}"

    rc = recall_at_k(retrieved_ids, query.relevant_chunk_ids, k=top_k)

    return QueryResult(
        query=query, retrieved_ids=retrieved_ids, recall_at_k=rc,
        latency_ms=latency_ms, status_code=last_status, error=error,
    )


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------

def print_results_table(
    results: list[QueryResult],
    top_k: int,
    repeats: int,
    base_url: str,
    use_color: bool,
    mode: str,
) -> None:

    def c(text, *codes):
        return _colorize(str(text), *codes, use_color=use_color)

    W_IDX    = 3
    W_LABEL  = 22
    W_TOPIC  = 12
    W_RC     = 10
    W_LAT    = 12
    W_STATUS = 7
    W_HITS   = 14
    TOTAL_W  = W_IDX + W_LABEL + W_TOPIC + W_RC + W_LAT + W_STATUS + W_HITS + 13

    rule     = "─" * TOTAL_W
    dbl_rule = "═" * TOTAL_W

    def row(idx, label, topic, rc, lat, status, hits):
        return (
            f"  {str(idx):<{W_IDX}}  "
            f"{label:<{W_LABEL}}  "
            f"{topic:<{W_TOPIC}}  "
            f"{rc:>{W_RC}}  "
            f"{lat:>{W_LAT}}  "
            f"{str(status):>{W_STATUS}}  "
            f"{hits:<{W_HITS}}"
        )

    header = row("#", "Query label", "Topic", f"Recall@{top_k}", "Latency (ms)", "HTTP", "Hits/Relevant")

    print()
    print(c("╔" + dbl_rule + "╗", _C.BOLD))
    title_str = f"  CSAI415 D2 — /search Evaluation  [{mode} mode]  ({base_url})"
    print(c("║" + title_str.center(TOTAL_W) + "║", _C.BOLD))
    print(c("╚" + dbl_rule + "╝", _C.BOLD))
    print()
    print(c(rule, _C.DIM))
    print(c(header, _C.BOLD))
    print(c(rule, _C.DIM))

    latencies_ok: list[float] = []
    recalls_ok:   list[float] = []

    for i, res in enumerate(results, start=1):
        label  = res.query.label[:W_LABEL]
        topic  = res.query.topic[:W_TOPIC]
        status = res.status_code

        if res.error:
            rc_str   = "ERROR"
            lat_str  = "—"
            hits_str = res.error[:W_HITS]
            color    = _C.RED
        else:
            rc_val   = res.recall_at_k
            rc_str   = f"{rc_val:.3f}"
            lat_str  = f"{res.latency_ms:.1f}"
            n_hits   = len(set(res.retrieved_ids[:top_k]) & set(res.query.relevant_chunk_ids))
            n_rel    = len(res.query.relevant_chunk_ids)
            hits_str = f"{n_hits} / {n_rel}"
            color    = _C.GREEN if rc_val >= 1.0 else (_C.YELLOW if rc_val >= 0.5 else _C.RED)
            latencies_ok.append(res.latency_ms)
            recalls_ok.append(rc_val)

        print(c(row(i, label, topic, rc_str, lat_str, status, hits_str), color))

    print(c(rule, _C.DIM))

    n_ok     = len(recalls_ok)
    n_err    = len(results) - n_ok
    mean_rc  = statistics.mean(recalls_ok)   if recalls_ok   else 0.0
    mean_lat = statistics.mean(latencies_ok) if latencies_ok else 0.0
    p95_lat  = (
        sorted(latencies_ok)[int(len(latencies_ok) * 0.95)]
        if len(latencies_ok) >= 2 else
        (latencies_ok[0] if latencies_ok else 0.0)
    )

    def rc_color(val):
        return _C.GREEN if val >= 0.8 else (_C.YELLOW if val >= 0.5 else _C.RED)

    print()
    print(c(f"  Queries       : {len(results)}  ({n_ok} ok, {n_err} errors)", _C.BOLD))
    print(c(f"  Ground-truth  : {mode} mode", _C.DIM))
    print(c(f"  top_k / reps  : {top_k} / {repeats}", _C.DIM))
    print()
    print(c(f"  Mean Recall@{top_k} : ", _C.BOLD) + c(f"{mean_rc:.3f}", _C.BOLD, rc_color(mean_rc)))
    print(c(f"  Mean latency  : {mean_lat:.1f} ms", _C.BOLD))
    print(c(f"  p95  latency  : {p95_lat:.1f} ms", _C.BOLD))
    print()

    if n_err > 0:
        verdict = c(f"  ✗  {n_err} query(s) failed — check server logs", _C.BOLD, _C.RED)
    elif mean_rc >= 0.8:
        verdict = c(f"  ✓  Recall looks healthy (≥ 0.8 mean)", _C.BOLD, _C.GREEN)
    elif mean_rc >= 0.5:
        verdict = c(f"  ⚠  Recall is moderate — check BM25/dense weights or embeddings", _C.BOLD, _C.YELLOW)
    else:
        verdict = c(f"  ✗  Recall is low — embeddings may not be indexed yet", _C.BOLD, _C.RED)

    print(verdict)
    print(c(rule, _C.DIM))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CSAI415 D2 — evaluate the FastAPI /search endpoint.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-url", default=os.getenv("API_BASE_URL", "http://localhost:8000"))
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--bm25-weight", type=float, default=0.4)
    p.add_argument("--dense-weight", type=float, default=0.6)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout", type=float, default=10.0)
    p.add_argument("--no-color", action="store_true")
    # FIX #7: new flags for mongo ground-truth mode
    p.add_argument(
        "--mongo-gt", action="store_true",
        help="Discover real chunk IDs from MongoDB instead of using seed.py IDs. "
             "Use this after running python ingest.py on your real PDF corpus.",
    )
    p.add_argument(
        "--mongo-uri", default=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        help="MongoDB URI — used only with --mongo-gt.",
    )
    p.add_argument(
        "--mongo-db", default=os.getenv("MONGO_DB", "csai415"),
        help="MongoDB database name — used only with --mongo-gt.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    use_color = not args.no_color and sys.stdout.isatty()

    def c(text, *codes):
        return _colorize(str(text), *codes, use_color=use_color)

    # Determine mode and build query list
    if args.mongo_gt:
        mode = "mongo"
        queries = _discover_ground_truth(
            QUERIES_SEED,
            mongo_uri=args.mongo_uri,
            mongo_db=args.mongo_db,
        )
    else:
        mode = "seed"
        queries = QUERIES_SEED
        print(c(
            "\n  [info] Running in seed mode — chunk IDs match seed.py output.\n"
            "         Use --mongo-gt after running ingest.py for real corpus evaluation.",
            _C.DIM,
        ))

    print(c(f"\nCSAI415 D2 — eval_search.py", _C.BOLD))
    print(c(f"  target  : {args.base_url}", _C.DIM))
    print(c(f"  mode    : {mode}", _C.DIM))
    print(c(f"  queries : {len(queries)}", _C.DIM))
    print(c(f"  top_k   : {args.top_k}    repeats: {args.repeats}", _C.DIM))

    print(c("\n  Checking /health …", _C.DIM), end="", flush=True)
    check_health(args.base_url, timeout=args.timeout)
    print(c("  OK", _C.GREEN))

    results: list[QueryResult] = []
    print()
    for i, query in enumerate(queries, start=1):
        label = query.label or query.query_text[:30]
        print(c(f"  [{i:>2}/{len(queries)}]", _C.DIM) + f"  {label:<30}", end="", flush=True)
        res = run_query(
            base_url=args.base_url, query=query, top_k=args.top_k,
            bm25_weight=args.bm25_weight, dense_weight=args.dense_weight,
            timeout=args.timeout, repeats=args.repeats,
        )
        results.append(res)
        if res.error:
            print(c(f"  ERROR: {res.error}", _C.RED))
        else:
            rc_col = _C.GREEN if res.recall_at_k >= 1.0 else (_C.YELLOW if res.recall_at_k >= 0.5 else _C.RED)
            print(c(f"  Recall@{args.top_k}={res.recall_at_k:.3f}", rc_col) + c(f"  {res.latency_ms:.0f} ms", _C.DIM))

    print_results_table(results, top_k=args.top_k, repeats=args.repeats,
                        base_url=args.base_url, use_color=use_color, mode=mode)


if __name__ == "__main__":
    main()
