#!/usr/bin/env python3
"""evaluate_graphrag.py

Run the D3 GraphRAG executor on every question in gold_qa.json and score the
answers with RAGAS faithfulness and answer/response relevancy metrics.

Expected project layout
-----------------------
Place this file in the same folder as:
    - graphrag_executor.py
    - graph_selector.py
    - safety.py
    - retriever_bridge.py  (from Deliverable 2, needed for vector/hybrid mode)
    - gold_qa.json

Basic usage
-----------
    python evaluate_graphrag.py --gold gold_qa.json --out d3_ragas_results.json

Run a specific ablation mode:
    python evaluate_graphrag.py --mode vector
    python evaluate_graphrag.py --mode graph
    python evaluate_graphrag.py --mode hybrid

Run all three D3 ablations in one command:
    python evaluate_graphrag.py --modes vector graph hybrid --out d3_ablation_ragas.json

RAGAS usually needs an evaluator LLM/embedding backend. The simplest setup is:
    pip install ragas datasets langchain-openai openai
    export OPENAI_API_KEY="your_key_here"       # macOS/Linux
    setx OPENAI_API_KEY "your_key_here"         # Windows PowerShell/CMD

Notes
-----
- The GraphRAG executor returns evidence chunks in result.blended. This script
  passes those chunk texts to RAGAS as retrieved contexts.
- The gold answer from gold_qa.json is passed as ground_truth/reference.
- The default Mongo database is set to csai415 because Deliverable 2 commonly
  used that DB name. Override with --mongo-db if your .env uses another name.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any


DEFAULT_MODES = ["hybrid"]
VALID_MODES = ["vector", "graph", "hybrid"]


def p95(values: list[float]) -> float:
    """Return p95 latency for a small evaluation list."""
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    return float(statistics.quantiles(values, n=20)[18])


def mean_numeric(rows: list[dict[str, Any]], key: str) -> float | None:
    vals: list[float] = []
    for row in rows:
        value = row.get(key)
        try:
            if value is not None and value == value:  # filters NaN without numpy
                vals.append(float(value))
        except (TypeError, ValueError):
            continue
    if not vals:
        return None
    return round(sum(vals) / len(vals), 4)


def first_numeric(row: dict[str, Any], keys: list[str]) -> float | None:
    """Return the first numeric metric value found across possible RAGAS key names."""
    for key in keys:
        value = row.get(key)
        try:
            if value is not None and value == value:  # filters NaN without numpy
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def format_score(value: Any) -> str:
    """Format score/latency values for the terminal summary table."""
    if value is None:
        return "-"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def summary_value(summary: dict[str, Any], aliases: list[str]) -> float | None:
    """Read one summary metric using common aliases across RAGAS versions."""
    for key in aliases:
        value = summary.get(key)
        try:
            if value is not None and value == value:
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def print_summary_table(summaries: list[dict[str, Any]]) -> None:
    """Print a clean final table for the D3 report/demo."""
    if not summaries:
        print("No summaries available.")
        return

    faithfulness_aliases = [
        "mean_faithfulness",
        "faithfulness",
    ]
    relevance_aliases = [
        "mean_answer_relevancy",
        "mean_answer_relevance",
        "mean_response_relevancy",
        "mean_response_relevance",
        "answer_relevancy",
        "answer_relevance",
        "response_relevancy",
        "response_relevance",
    ]

    headers = [
        "Mode",
        "Questions",
        "Faithfulness",
        "Answer relevance",
        "p95 latency (s)",
        "Mean latency (s)",
        "Citation rate",
        "Context rate",
    ]

    table_rows: list[list[str]] = []
    for summary in summaries:
        row = [
            str(summary.get("mode", "-")),
            str(summary.get("num_questions", "-")),
            format_score(summary_value(summary, faithfulness_aliases)),
            format_score(summary_value(summary, relevance_aliases)),
            format_score(summary.get("latency_p95_seconds")),
            format_score(summary.get("latency_mean_seconds")),
            format_score(summary.get("citation_availability_rate")),
            format_score(summary.get("context_availability_rate")),
        ]
        table_rows.append(row)

    widths = [len(h) for h in headers]
    for row in table_rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]

    def line(char: str = "-") -> str:
        return "+" + "+".join(char * (width + 2) for width in widths) + "+"

    def render_row(cells: list[str]) -> str:
        return "|" + "|".join(f" {cell:<{width}} " for cell, width in zip(cells, widths)) + "|"

    print(line("="))
    print(render_row(headers))
    print(line("="))
    for row in table_rows:
        print(render_row(row))
    print(line("="))


def resolve_path(path_text: str | None, default_name: str, base_dir: Path) -> Path:
    """Resolve a path from cwd first, then from this script's directory."""
    candidate = Path(path_text or default_name)
    if candidate.is_absolute():
        return candidate
    cwd_candidate = Path.cwd() / candidate
    if cwd_candidate.exists():
        return cwd_candidate
    return base_dir / candidate


def load_gold_qa(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Gold QA file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        # Supports either {"items": [...]} or {"questions": [...]} style files.
        rows = data.get("items") or data.get("questions") or data.get("gold_qa") or []
    else:
        raise ValueError("gold_qa.json must contain a list or a dict containing a list.")

    clean: list[dict[str, Any]] = []
    for i, item in enumerate(rows, start=1):
        if not isinstance(item, dict):
            continue
        question = item.get("question") or item.get("query")
        answer = item.get("answer") or item.get("ground_truth") or item.get("reference")
        if not question or not answer:
            raise ValueError(
                f"Gold QA item #{i} must include at least 'question' and 'answer'."
            )
        clean.append(item)

    if not clean:
        raise ValueError("No valid QA items found in gold_qa.json.")
    return clean


def compact_contexts(blended: list[dict[str, Any]], max_contexts: int) -> list[str]:
    """Extract retrieved chunk texts for RAGAS contexts."""
    contexts: list[str] = []
    seen: set[str] = set()
    for chunk in blended or []:
        text = " ".join(str(chunk.get("text") or "").split())
        if not text or text in seen:
            continue
        seen.add(text)
        contexts.append(text)
        if len(contexts) >= max_contexts:
            break
    return contexts


def run_graphrag(
    gold_qa: list[dict[str, Any]],
    mode: str,
    top_k: int,
    top_papers: int,
    chunks_per_paper: int,
    alpha: float | None,
    rerank: bool,
    max_contexts: int,
    verbose: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run GraphRAG and return (ragas_samples, raw_execution_rows)."""
    from graphrag_executor import GraphRAGExecutor

    ragas_samples: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []

    with GraphRAGExecutor(verbose=verbose, rerank=rerank) as executor:
        for idx, item in enumerate(gold_qa, start=1):
            question = str(item["question"])
            ground_truth = str(
                item.get("answer") or item.get("ground_truth") or item.get("reference")
            )

            start = time.perf_counter()
            result = executor.answer(
                query=question,
                top_k=top_k,
                top_papers=top_papers,
                chunks_per_paper=chunks_per_paper,
                alpha=alpha,
                mode=mode,
                rerank=rerank,
            )
            latency = time.perf_counter() - start

            contexts = compact_contexts(result.blended, max_contexts=max_contexts)

            # RAGAS legacy evaluate() accepts these fields:
            # question, answer, contexts, ground_truth.
            ragas_samples.append(
                {
                    "question": question,
                    "answer": result.answer,
                    "contexts": contexts,
                    "ground_truth": ground_truth,
                }
            )

            raw_rows.append(
                {
                    "id": item.get("id") or f"Q{idx:03d}",
                    "question": question,
                    "gold_answer": ground_truth,
                    "generated_answer": result.answer,
                    "mode_requested": mode,
                    "mode_used": result.mode,
                    "latency_seconds": round(latency, 4),
                    "context_count": len(contexts),
                    "citation_count": len(result.citations),
                    "citations": result.citations,
                    "selected_papers": result.selected_papers,
                    "selected_topics": result.selected_topics,
                    "selected_authors": result.selected_authors,
                    "warning": result.warning,
                    "expected_topic": item.get("expected_topic"),
                    "expected_papers": item.get("expected_papers"),
                    "expected_titles": item.get("expected_titles"),
                    "required_keywords": item.get("required_keywords"),
                }
            )

            print(
                f"[{mode}] {idx}/{len(gold_qa)} | "
                f"contexts={len(contexts)} citations={len(result.citations)} "
                f"latency={latency:.2f}s | {question}"
            )

    return ragas_samples, raw_rows


def get_ragas_metrics() -> list[Any]:
    """Import RAGAS metrics across common RAGAS versions.

    Most course environments use the legacy metric objects:
        from ragas.metrics import faithfulness, answer_relevancy

    Newer RAGAS versions also expose collection/class metrics. This function
    keeps the script usable across both without changing the student code.
    """
    import_errors: list[str] = []

    try:
        from ragas.metrics import answer_relevancy, faithfulness

        return [faithfulness, answer_relevancy]
    except Exception as exc:  # noqa: BLE001
        import_errors.append(f"legacy ragas.metrics objects failed: {exc}")

    try:
        from ragas.metrics import Faithfulness, ResponseRelevancy

        return [Faithfulness(), ResponseRelevancy()]
    except Exception as exc:  # noqa: BLE001
        import_errors.append(f"ragas.metrics classes failed: {exc}")

    try:
        from ragas.metrics.collections import Faithfulness, ResponseRelevancy

        return [Faithfulness(), ResponseRelevancy()]
    except Exception as exc:  # noqa: BLE001
        import_errors.append(f"ragas.metrics.collections classes failed: {exc}")

    raise ImportError(
        "Could not import RAGAS faithfulness/relevancy metrics. "
        "Install/upgrade ragas, for example: pip install -U ragas datasets.\n"
        + "\n".join(import_errors)
    )


def run_ragas(ragas_samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compute RAGAS metrics and return per-row metric rows + summary dict."""
    try:
        from datasets import Dataset
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "Missing dependency 'datasets'. Install it with: pip install datasets"
        ) from exc

    try:
        from ragas import evaluate
    except Exception as exc:  # noqa: BLE001
        raise ImportError("Missing dependency 'ragas'. Install it with: pip install ragas") from exc

    metrics = get_ragas_metrics()
    dataset = Dataset.from_list(ragas_samples)

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        raise_exceptions=False,
        show_progress=True,
    )

    metric_rows: list[dict[str, Any]]
    summary: dict[str, Any] = {}

    if hasattr(result, "to_pandas"):
        df = result.to_pandas()
        metric_rows = df.to_dict(orient="records")
        for col in df.columns:
            try:
                if str(df[col].dtype).startswith(("float", "int")):
                    summary[col] = round(float(df[col].mean()), 4)
            except Exception:  # noqa: BLE001
                pass
    elif hasattr(result, "scores"):
        metric_rows = list(getattr(result, "scores") or [])
    elif isinstance(result, dict):
        # Some versions return only aggregate scores.
        metric_rows = []
        summary = dict(result)
    else:
        metric_rows = []
        summary = {"raw_result": str(result)}

    # Add fallback summary values from per-row records.
    for key in [
        "faithfulness",
        "answer_relevancy",
        "answer_relevance",
        "response_relevancy",
        "response_relevance",
    ]:
        avg = mean_numeric(metric_rows, key)
        if avg is not None:
            summary[f"mean_{key}"] = avg

    return metric_rows, summary


def merge_rows(raw_rows: list[dict[str, Any]], metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Join GraphRAG execution metadata with RAGAS metric output by row order."""
    merged: list[dict[str, Any]] = []
    for i, raw in enumerate(raw_rows):
        out = dict(raw)
        if i < len(metric_rows):
            metrics = dict(metric_rows[i])
            # Avoid duplicating very long fields from RAGAS output.
            metrics.pop("answer", None)
            metrics.pop("contexts", None)
            metrics.pop("ground_truth", None)
            metrics.pop("question", None)
            out.update(metrics)
        merged.append(out)
    return merged


def build_summary(rows: list[dict[str, Any]], ragas_summary: dict[str, Any], mode: str) -> dict[str, Any]:
    latencies = [float(r["latency_seconds"]) for r in rows if r.get("latency_seconds") is not None]
    total = len(rows)
    citation_hits = sum(1 for r in rows if int(r.get("citation_count") or 0) > 0)
    context_hits = sum(1 for r in rows if int(r.get("context_count") or 0) > 0)

    summary = {
        "mode": mode,
        "num_questions": total,
        "latency_p95_seconds": round(p95(latencies), 4),
        "latency_mean_seconds": round(sum(latencies) / total, 4) if total else 0.0,
        "citation_availability_rate": round(citation_hits / total, 4) if total else 0.0,
        "context_availability_rate": round(context_hits / total, 4) if total else 0.0,
    }

    # Prefer the per-row merged metrics because names differ across RAGAS versions.
    for key in [
        "faithfulness",
        "answer_relevancy",
        "answer_relevance",
        "response_relevancy",
        "response_relevance",
    ]:
        avg = mean_numeric(rows, key)
        if avg is not None:
            summary[f"mean_{key}"] = avg

    # Preserve aggregate values returned by RAGAS too.
    for key, value in ragas_summary.items():
        if key not in summary:
            try:
                summary[key] = round(float(value), 4)
            except Exception:  # noqa: BLE001
                summary[key] = value

    return summary


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    if str(base_dir) not in sys.path:
        sys.path.insert(0, str(base_dir))

    parser = argparse.ArgumentParser(
        description="Evaluate D3 GraphRAG answers with RAGAS faithfulness and relevancy."
    )
    parser.add_argument("--gold", default="gold_qa.json", help="Path to gold_qa.json")
    parser.add_argument("--out", default="d3_ragas_results.json", help="Output JSON path")
    parser.add_argument(
        "--mode",
        choices=VALID_MODES,
        default="hybrid",
        help="Single GraphRAG mode to evaluate.",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=VALID_MODES,
        default=None,
        help="Evaluate multiple modes, e.g. --modes vector graph hybrid",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--top-papers", type=int, default=5)
    parser.add_argument("--chunks-per-paper", type=int, default=3)
    parser.add_argument("--max-contexts", type=int, default=5)
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="BM25 weight for the D2 hybrid retriever. Uses retriever default if omitted.",
    )
    parser.add_argument(
        "--no-rerank",
        action="store_true",
        help="Disable GraphRAG executor cross-encoder reranking.",
    )
    parser.add_argument(
        "--mongo-db",
        default="csai415",
        help="MongoDB database name. Default aligns with Deliverable 2.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce executor logs.")
    args = parser.parse_args()

    # Must be set before GraphRAGExecutor is imported/constructed.
    os.environ.setdefault("MONGO_DB", args.mongo_db)

    gold_path = resolve_path(args.gold, "gold_qa.json", base_dir)
    out_path = resolve_path(args.out, "d3_ragas_results.json", base_dir)
    modes = args.modes if args.modes else [args.mode]

    gold_qa = load_gold_qa(gold_path)

    final: dict[str, Any] = {
        "config": {
            "gold_file": str(gold_path),
            "modes": modes,
            "top_k": args.top_k,
            "top_papers": args.top_papers,
            "chunks_per_paper": args.chunks_per_paper,
            "max_contexts": args.max_contexts,
            "alpha": args.alpha,
            "rerank": not args.no_rerank,
            "mongo_db": os.environ.get("MONGO_DB"),
        },
        "summaries": [],
        "results_by_mode": {},
    }

    for mode in modes:
        print(f"\n========== Running GraphRAG mode: {mode} ==========")
        ragas_samples, raw_rows = run_graphrag(
            gold_qa=gold_qa,
            mode=mode,
            top_k=args.top_k,
            top_papers=args.top_papers,
            chunks_per_paper=args.chunks_per_paper,
            alpha=args.alpha,
            rerank=not args.no_rerank,
            max_contexts=args.max_contexts,
            verbose=not args.quiet,
        )

        print(f"\n========== Running RAGAS for mode: {mode} ==========")
        metric_rows, ragas_summary = run_ragas(ragas_samples)
        merged_rows = merge_rows(raw_rows, metric_rows)
        summary = build_summary(merged_rows, ragas_summary, mode)

        final["summaries"].append(summary)
        final["results_by_mode"][mode] = {
            "summary": summary,
            "rows": merged_rows,
        }

    out_path.write_text(json.dumps(final, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n========== Evaluation summary ==========")
    print_summary_table(final["summaries"])
    print("\nDetailed summary JSON:")
    print(json.dumps(final["summaries"], indent=2, ensure_ascii=False))
    print(f"\nSaved full results to: {out_path}")


if __name__ == "__main__":
    main()
