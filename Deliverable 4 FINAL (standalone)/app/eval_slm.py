"""eval_slm.py — Deliverable 4 final quality/latency table: zero-shot vs tuned.

Runs the gold Q/A set through the GraphRAG answer step with each SLM backend and
reports the quality + latency deltas the brief asks for:

    backend     faithfulness   answer_relevance   mean_ms   p95_ms   cache_hit
    extractive  ...            ...                ...       ...      ...
    base        ...            ...                ...       ...      ...   (zero-shot)
    tuned       ...            ...                ...       ...      ...   (PEFT/QLoRA)

Context source
--------------
By default eval runs **offline** (no MongoDB / Qdrant / Neo4j needed): a tiny
in-memory keyword retriever scores the real corpus abstracts and feeds the top-k
as grounded context.  This makes the comparison reproducible on any laptop.
Pass --use-graphrag to instead pull contexts from the full D2/D3 stack when the
services are running.

Metrics reuse ablation.py's deterministic lexical scorer (faithfulness =
answer-vs-context overlap; relevance = answer-vs-gold + required-keyword
coverage), so no API key is required.  The same numbers can be regenerated with
RAGAS via evaluate_graphrag.py when credentials are available.

Usage
-----
  python eval_slm.py                                  # extractive vs base vs tuned (offline)
  python eval_slm.py --backends extractive base tuned --base-model sshleifer/tiny-gpt2
  python eval_slm.py --use-graphrag                   # contexts from D2/D3 stack
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from config import ARTIFACTS_DIR, DATA_DIR, REPORTS_DIR
from ablation import lexical_metrics, tokenize          # reuse D3 deterministic metrics
from slm import SLMConfig, AnswerGenerator


# ---------------------------------------------------------------------------
# Offline mini-retriever over real corpus abstracts (no services required)
# ---------------------------------------------------------------------------

class AbstractRetriever:
    """Keyword-overlap retriever over the downloaded corpus abstracts."""

    def __init__(self) -> None:
        self.docs: list[dict] = []
        meta_path = DATA_DIR / "corpus_metadata.json"
        if meta_path.exists():
            from build_qa_dataset import extract_abstract
            from config import PDF_DIR
            for p in json.loads(meta_path.read_text(encoding="utf-8")):
                pdf = PDF_DIR / p["pdf_filename"]
                abstract = extract_abstract(pdf) if pdf.exists() else p["title"]
                self.docs.append({
                    "chunk_id": f"{p['paper_id'].lower()}_abs",
                    "title": p["title"],
                    "page_range": "p. 1",
                    "text": abstract or p["title"],
                    "tokens": tokenize(f"{p['title']} {abstract} {p['topics']}"),
                })

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        q = tokenize(query)
        scored = []
        for d in self.docs:
            overlap = len(q & d["tokens"])
            if overlap:
                scored.append((overlap, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in scored[:top_k]] or [self.docs[0]] if self.docs else []


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    if len(values) < 2:
        return values[0]
    return statistics.quantiles(values, n=20)[18]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_backend(backend: str, gold: list[dict], retriever, base_model: str,
                use_graphrag: bool, verbose: bool) -> dict:
    gen = AnswerGenerator(SLMConfig(backend=backend, base_model=base_model), verbose=verbose)

    executor = None
    if use_graphrag:
        from graphrag_executor import GraphRAGExecutor
        executor = GraphRAGExecutor(verbose=False, rerank=False)

    # Warm up the model once (untimed) so lazy load/compile doesn't pollute p95.
    if backend != "extractive" and gold:
        first_q = gold[0].get("question") or gold[0].get("query")
        warm_ctx = (retriever.search(first_q, top_k=3) if retriever is not None else [])
        gen.generate(first_q, warm_ctx)

    faith, rel, lats = [], [], []
    cache_hits = 0
    effective_backend = backend
    rows = []
    for item in gold:
        q = item.get("question") or item.get("query")
        gold_a = item.get("answer") or ""
        req_kw = item.get("required_keywords") or []

        if executor is not None:
            res = executor.answer(q, top_k=4, mode="hybrid")
            contexts = res.blended or []
        else:
            contexts = retriever.search(q, top_k=3)

        t0 = time.perf_counter()
        out = gen.generate(q, contexts)
        lat = (time.perf_counter() - t0) * 1000
        effective_backend = out["backend"]
        if out["cached"]:
            cache_hits += 1

        m = lexical_metrics(q, out["answer"], gold_a,
                            [c.get("text", "") for c in contexts], req_kw)
        faith.append(m["faithfulness"])
        rel.append(m["answer_relevance"])
        lats.append(lat)
        rows.append({"question": q, "faithfulness": m["faithfulness"],
                     "answer_relevance": m["answer_relevance"], "latency_ms": round(lat, 2),
                     "cached": out["cached"]})

    if executor is not None:
        executor.close()

    n = len(gold)
    return {
        "requested_backend": backend,
        "effective_backend": effective_backend,
        "num_questions": n,
        "faithfulness": round(sum(faith) / n, 4) if n else 0.0,
        "answer_relevance": round(sum(rel) / n, 4) if n else 0.0,
        "mean_ms": round(sum(lats) / n, 2) if n else 0.0,
        "p95_ms": round(p95(lats), 2),
        "cache_hits": cache_hits,
        "rows": rows,
    }


def markdown_table(summaries: list[dict]) -> str:
    headers = ["Backend", "Faithfulness", "Answer relevance", "Mean ms", "p95 ms", "Cache hits"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for s in summaries:
        label = s["effective_backend"]
        if s["effective_backend"] != s["requested_backend"]:
            label = f"{s['requested_backend']}->{s['effective_backend']}"
        lines.append("| " + " | ".join([
            label, f"{s['faithfulness']:.4f}", f"{s['answer_relevance']:.4f}",
            f"{s['mean_ms']:.1f}", f"{s['p95_ms']:.1f}", str(s["cache_hits"]),
        ]) + " |")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="D4 final eval: extractive vs base vs tuned SLM.")
    ap.add_argument("--gold", default=str(Path(__file__).resolve().parent / "gold_qa.json"))
    ap.add_argument("--backends", nargs="+", default=["extractive", "base", "tuned"])
    ap.add_argument("--base-model", default=None, help="Override SLM base model (e.g. for CPU smoke).")
    ap.add_argument("--use-graphrag", action="store_true", help="Pull contexts from D2/D3 stack.")
    ap.add_argument("--out", default=str(Path(ARTIFACTS_DIR) / "slm_eval.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    if isinstance(gold, dict):
        gold = gold.get("items") or gold.get("questions") or []
    base_model = args.base_model or None

    retriever = None if args.use_graphrag else AbstractRetriever()

    summaries = []
    for backend in args.backends:
        print(f"\n========== eval backend: {backend} ==========")
        kw = {}
        if base_model:
            kw["base_model"] = base_model
        # AnswerGenerator reads base_model from config by default; override below.
        s = run_backend(backend, gold, retriever,
                        base_model=base_model or SLMConfig().base_model,
                        use_graphrag=args.use_graphrag, verbose=args.verbose)
        summaries.append(s)
        print(f"  faithfulness={s['faithfulness']:.4f}  relevance={s['answer_relevance']:.4f}  "
              f"mean={s['mean_ms']:.1f}ms  p95={s['p95_ms']:.1f}ms  cache_hits={s['cache_hits']}")

    table = markdown_table(summaries)
    print("\n========== D4 SLM comparison ==========")
    print(table)

    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summaries": summaries}, indent=2), encoding="utf-8")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "D4_slm_eval.md").write_text(
        "# D4 — SLM Integration Eval (zero-shot vs tuned)\n\n"
        f"Context source: {'GraphRAG (D2/D3 stack)' if args.use_graphrag else 'offline abstract retriever'}\n\n"
        f"{table}\n", encoding="utf-8")
    print(f"\nSaved: {args.out}")
    print(f"Saved: {REPORTS_DIR / 'D4_slm_eval.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
