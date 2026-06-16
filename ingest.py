"""eval_search.py — CSAI415 D2 /search endpoint evaluator.

Fires 10 sample queries at the FastAPI /search endpoint, measures
Recall@5 and latency per query, and prints a clean summary table.

The ground-truth relevant chunk IDs come from the three papers inserted
by seed.py (P001 Attention, P002 BERT, P003 RAG).  If you have run the
real ingest pipeline against your own PDFs, swap in your own QUERIES list
at the bottom of this file.

Usage
-----
    python eval_search.py                          # defaults (localhost:8000)
    python eval_search.py --base-url http://localhost:8000
    python eval_search.py --top-k 5 --repeats 3
    python eval_search.py --no-color               # plain text (for logs/CI)

Requirements
------------
    pip install requests
    (FastAPI server must be running: uvicorn api:app --reload)
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

try:
    import requests
except ImportError:
    sys.exit("requests not found. Run: pip install requests")


# ---------------------------------------------------------------------------
# ANSI colours (disabled via --no-color or when stdout is not a TTY)
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
# Query + ground-truth definitions
# ---------------------------------------------------------------------------

@dataclass
class EvalQuery:
    """One evaluation query with its expected relevant chunk IDs."""
    query_text:          str
    relevant_chunk_ids:  list[str]          # ground-truth set
    label:               str = ""           # short display label
    topic:               str = ""           # for the table


# Ground truth matches the chunk_ids inserted by seed.py.
# Each paper has 2 chunks: p00X_c0 and p00X_c1.
QUERIES: list[EvalQuery] = [
    # ── Transformers / Attention ──────────────────────────────────────────
    EvalQuery(
        query_text="transformer architecture attention mechanism",
        relevant_chunk_ids=["p001_c0", "p001_c1"],
        label="Transformer attention",
        topic="Transformers",
    ),
    EvalQuery(
        query_text="self-attention network dispensing with recurrence",
        relevant_chunk_ids=["p001_c0"],
        label="No recurrence",
        topic="Transformers",
    ),
    EvalQuery(
        query_text="multi-head attention representation subspaces",
        relevant_chunk_ids=["p001_c1"],
        label="Multi-head attention",
        topic="Transformers",
    ),
    # ── BERT / Pre-training ───────────────────────────────────────────────
    EvalQuery(
        query_text="BERT bidirectional pre-training language model",
        relevant_chunk_ids=["p002_c0", "p002_c1"],
        label="BERT pre-training",
        topic="BERT",
    ),
    EvalQuery(
        query_text="pre-train deep representations from unlabelled text",
        relevant_chunk_ids=["p002_c0"],
        label="Unlabelled text",
        topic="BERT",
    ),
    EvalQuery(
        query_text="fine-tuning NLP tasks state-of-the-art results",
        relevant_chunk_ids=["p002_c1"],
        label="Fine-tuning BERT",
        topic="BERT",
    ),
    # ── RAG / Retrieval ───────────────────────────────────────────────────
    EvalQuery(
        query_text="retrieval augmented generation knowledge intensive tasks",
        relevant_chunk_ids=["p003_c0", "p003_c1"],
        label="RAG overview",
        topic="RAG",
    ),
    EvalQuery(
        query_text="dense retriever fetch documents seq2seq generation",
        relevant_chunk_ids=["p003_c0"],
        label="Dense retriever",
        topic="RAG",
    ),
    EvalQuery(
        query_text="open domain question answering factual grounding",
        relevant_chunk_ids=["p003_c1"],
        label="Open-domain QA",
        topic="RAG",
    ),
    # ── Cross-topic ───────────────────────────────────────────────────────
    EvalQuery(
        query_text="NLP deep learning pre-trained models transformers BERT",
        relevant_chunk_ids=["p001_c0", "p001_c1", "p002_c0", "p002_c1"],
        label="NLP broad query",
        topic="Cross-topic",
    ),
]


# ---------------------------------------------------------------------------
# Per-query result
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query:          EvalQuery
    retrieved_ids:  list[str]
    recall_at_k:    float
    latency_ms:     float
    status_code:    int
    error:          Optional[str] = None


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
    """Abort early with a clear message if the server is not up."""
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
    """Call /search, time it over `repeats` passes, return a QueryResult."""
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

    for rep in range(repeats):
        t0 = time.perf_counter()
        try:
            resp = requests.get(f"{base_url}/search", params=params, timeout=timeout)
            elapsed_ms = (time.perf_counter() - t0) * 1_000
            latencies.append(elapsed_ms)
            last_response = resp
            last_status   = resp.status_code
        except requests.RequestException as exc:
            error = str(exc)
            return QueryResult(
                query=query,
                retrieved_ids=[],
                recall_at_k=0.0,
                latency_ms=0.0,
                status_code=0,
                error=error,
            )

    # Use median latency across repeats — more robust than mean for small n
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
        query=query,
        retrieved_ids=retrieved_ids,
        recall_at_k=rc,
        latency_ms=latency_ms,
        status_code=last_status,
        error=error,
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
) -> None:

    def c(text, *codes):
        return _colorize(str(text), *codes, use_color=use_color)

    # ── column widths ────────────────────────────────────────────────────────
    W_IDX    = 3
    W_LABEL  = 22
    W_TOPIC  = 12
    W_RC     = 10
    W_LAT    = 12
    W_STATUS = 7
    W_HITS   = 14

    TOTAL_W = W_IDX + W_LABEL + W_TOPIC + W_RC + W_LAT + W_STATUS + W_HITS + 13

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
    print(c("║" + f"  CSAI415 D2 — /search Evaluation  ({base_url})".center(TOTAL_W) + "║", _C.BOLD))
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
            rc_str  = "ERROR"
            lat_str = "—"
            hits_str = res.error[:W_HITS]
            color = _C.RED
        else:
            rc_val  = res.recall_at_k
            rc_str  = f"{rc_val:.3f}"
            lat_str = f"{res.latency_ms:.1f}"
            n_hits  = len(set(res.retrieved_ids[:top_k]) & set(res.query.relevant_chunk_ids))
            n_rel   = len(res.query.relevant_chunk_ids)
            hits_str = f"{n_hits} / {n_rel}"

            if rc_val >= 1.0:
                color = _C.GREEN
            elif rc_val >= 0.5:
                color = _C.YELLOW
            else:
                color = _C.RED

            latencies_ok.append(res.latency_ms)
            recalls_ok.append(rc_val)

        line = row(i, label, topic, rc_str, lat_str, status, hits_str)
        print(c(line, color))

    print(c(rule, _C.DIM))

    # ── summary stats ────────────────────────────────────────────────────────
    n_ok     = len(recalls_ok)
    n_err    = len(results) - n_ok
    mean_rc  = statistics.mean(recalls_ok)   if recalls_ok  else 0.0
    mean_lat = statistics.mean(latencies_ok) if latencies_ok else 0.0
    p95_lat  = (
        sorted(latencies_ok)[int(len(latencies_ok) * 0.95)]
        if len(latencies_ok) >= 2 else
        (latencies_ok[0] if latencies_ok else 0.0)
    )
    min_lat  = min(latencies_ok) if latencies_ok else 0.0
    max_lat  = max(latencies_ok) if latencies_ok else 0.0

    def rc_color(val: float):
        if val >= 0.8:  return _C.GREEN
        if val >= 0.5:  return _C.YELLOW
        return _C.RED

    print()
    print(c(f"  Queries       : {len(results)}  ({n_ok} ok, {n_err} errors)", _C.BOLD))
    print(c(f"  top_k         : {top_k}", _C.DIM))
    print(c(f"  repeats/query : {repeats}  (median latency reported)", _C.DIM))
    print()
    print(c(f"  Mean Recall@{top_k} : ", _C.BOLD) +
          c(f"{mean_rc:.3f}", _C.BOLD, rc_color(mean_rc)))
    print(c(f"  Mean latency  : {mean_lat:.1f} ms", _C.BOLD))
    print(c(f"  p95  latency  : {p95_lat:.1f} ms", _C.BOLD))
    print(c(f"  Min  latency  : {min_lat:.1f} ms", _C.DIM))
    print(c(f"  Max  latency  : {max_lat:.1f} ms", _C.DIM))
    print()

    # Pass / warn / fail verdict
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
    p.add_argument(
        "--base-url",
        default=os.getenv("API_BASE_URL", "http://localhost:8000"),
        help="Base URL of the running FastAPI server.",
    )
    p.add_argument(
        "--top-k", type=int, default=5,
        help="Number of results to request per query (passed to /search).",
    )
    p.add_argument(
        "--bm25-weight", type=float, default=0.4,
        help="BM25 fusion weight passed to /search.",
    )
    p.add_argument(
        "--dense-weight", type=float, default=0.6,
        help="Dense fusion weight passed to /search.",
    )
    p.add_argument(
        "--repeats", type=int, default=3,
        help="Number of timed HTTP calls per query. Median latency is reported.",
    )
    p.add_argument(
        "--timeout", type=float, default=10.0,
        help="Per-request timeout in seconds.",
    )
    p.add_argument(
        "--no-color", action="store_true",
        help="Disable ANSI colour output (useful for CI or log files).",
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

    print(c(f"\nCSAI415 D2 — eval_search.py", _C.BOLD))
    print(c(f"  target  : {args.base_url}", _C.DIM))
    print(c(f"  queries : {len(QUERIES)}", _C.DIM))
    print(c(f"  top_k   : {args.top_k}    repeats: {args.repeats}", _C.DIM))

    # Health check before wasting time on queries
    print(c("\n  Checking /health …", _C.DIM), end="", flush=True)
    check_health(args.base_url, timeout=args.timeout)
    print(c("  OK", _C.GREEN))

    # Run all queries
    results: list[QueryResult] = []
    print()
    for i, query in enumerate(QUERIES, start=1):
        label = query.label or query.query_text[:30]
        print(
            c(f"  [{i:>2}/{len(QUERIES)}]", _C.DIM) +
            f"  {label:<30}",
            end="",
            flush=True,
        )
        res = run_query(
            base_url=args.base_url,
            query=query,
            top_k=args.top_k,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
            timeout=args.timeout,
            repeats=args.repeats,
        )
        results.append(res)

        if res.error:
            print(c(f"  ERROR: {res.error}", _C.RED))
        else:
            rc_col = _C.GREEN if res.recall_at_k >= 1.0 else (_C.YELLOW if res.recall_at_k >= 0.5 else _C.RED)
            print(
                c(f"  Recall@{args.top_k}={res.recall_at_k:.3f}", rc_col) +
                c(f"  {res.latency_ms:.0f} ms", _C.DIM)
            )

    print_results_table(
        results,
        top_k=args.top_k,
        repeats=args.repeats,
        base_url=args.base_url,
        use_color=use_color,
    )


if __name__ == "__main__":
    main()
