"""evaluation.py — lightweight D3 evaluation and ablation runner.

This provides a simple starting point for D3 metrics:
- latency p95
- graph-guided success rate
- citation availability rate

For full marks, you can later add RAGAS faithfulness and answer relevance here.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from graphrag_executor import GraphRAGExecutor


DEFAULT_GOLD_QA = [
    {"question": "What papers discuss transformers?", "expected_topic": "transformer"},
    {"question": "Which authors are related to retrieval?", "expected_topic": "retrieval"},
    {"question": "What papers are about attention?", "expected_topic": "attention"},
]


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=20)[18]


def run_eval(gold_qa: list[dict[str, Any]]) -> dict[str, Any]:
    latencies: list[float] = []
    citation_hits = 0
    graph_success = 0
    rows = []

    with GraphRAGExecutor(verbose=False) as executor:
        for item in gold_qa:
            question = item["question"]
            start = time.perf_counter()
            result = executor.answer(question)
            latency = time.perf_counter() - start
            latencies.append(latency)

            has_citation = len(result.citations) > 0
            if has_citation:
                citation_hits += 1
            if result.mode in {"graph_guided_with_chunks", "graph_selected"}:
                graph_success += 1

            rows.append(
                {
                    "question": question,
                    "mode": result.mode,
                    "latency_seconds": round(latency, 4),
                    "citation_count": len(result.citations),
                    "selected_papers": result.selected_papers,
                    "warning": result.warning,
                }
            )

    total = len(gold_qa)
    return {
        "summary": {
            "num_questions": total,
            "latency_p95_seconds": round(p95(latencies), 4),
            "citation_availability_rate": round(citation_hits / total, 3) if total else 0,
            "graph_success_rate": round(graph_success / total, 3) if total else 0,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lightweight D3 evaluation.")
    parser.add_argument("--gold", help="Path to gold QA JSON file", default=None)
    parser.add_argument("--out", help="Output JSON path", default="d3_eval_results.json")
    args = parser.parse_args()

    if args.gold:
        gold_qa = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    else:
        gold_qa = DEFAULT_GOLD_QA

    results = run_eval(gold_qa)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
