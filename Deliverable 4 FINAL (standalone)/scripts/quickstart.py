"""quickstart.py — one-command, offline end-to-end demo (no services needed).

Proves the agent runs on a bare laptop without MongoDB / Qdrant / Neo4j / GPU:

  1. ensure the curated Q/A dataset exists (built from the real corpus),
  2. answer a couple of questions with grounded, page-cited evidence using the
     offline abstract retriever + the extractive SLM backend,
  3. run the final D4 quality/latency table for the extractive backend.

Then it prints exactly how to bring up the full stack (Docker + ingest + graph)
and how to fine-tune + A/B the SLM.

Run:
    python scripts/quickstart.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
APP = ROOT / "app"
sys.path.insert(0, str(APP))


def banner(text: str) -> None:
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70)


def main() -> int:
    banner("PDF-Papers AI Agent — offline quickstart")

    # --- 0. corpus present? -------------------------------------------------
    meta = ROOT / "data" / "corpus_metadata.json"
    if not meta.exists():
        print("[1/3] Corpus metadata missing — downloading the real arXiv corpus...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "download_corpus.py")], check=False)
    else:
        n = len(json.loads(meta.read_text(encoding="utf-8")))
        print(f"[1/3] Corpus ready: {n} real arXiv PDFs in data/pdfs/")

    # --- 1. QA dataset ------------------------------------------------------
    qa = ROOT / "data" / "qa" / "qa_dataset.jsonl"
    if not qa.exists():
        print("      Building curated Q/A dataset from the corpus...")
        subprocess.run([sys.executable, str(APP / "build_qa_dataset.py")], cwd=str(APP), check=False)
    else:
        print(f"      Q/A dataset ready: {qa}")

    # --- 2. grounded demo answers (offline) ---------------------------------
    banner("[2/3] Grounded, page-cited answers (offline abstract retriever + extractive SLM)")
    from eval_slm import AbstractRetriever  # noqa: E402
    from slm import get_generator           # noqa: E402

    retriever = AbstractRetriever()
    gen = get_generator(backend="extractive")
    demo_questions = [
        "What is the Transformer architecture based on?",
        "What does LoRA do to adapt large language models?",
    ]
    for q in demo_questions:
        ctx = retriever.search(q, top_k=3)
        out = gen.generate(q, ctx)
        print(f"\nQ: {q}")
        print(out["answer"])

    # --- 3. final eval table (extractive backend) ---------------------------
    banner("[3/3] D4 quality/latency table (extractive backend, offline)")
    subprocess.run(
        [sys.executable, str(APP / "eval_slm.py"), "--backends", "extractive"],
        cwd=str(APP), check=False,
    )

    banner("Next steps")
    print(
        "Full stack (needs Docker):\n"
        "  docker compose up -d\n"
        "  cd app && python seed.py                       # smoke-test the stores\n"
        "  cd app && python ingest.py --pdf_dir ../data/pdfs\n"
        "  cd app && python build_graph.py --csv ../data/papers.csv\n"
        "  cd app && uvicorn api:app --reload             # then open http://localhost:8000/docs\n"
        "\nStreaming learner + AutoML (D1):\n"
        "  python run_d1.py\n"
        "\nFine-tune + A/B the SLM (D4):\n"
        "  pip install -r requirements-slm.txt\n"
        "  cd app && python train_slm.py                  # PEFT/QLoRA -> artifacts/slm_lora\n"
        "  cd app && python eval_slm.py --backends extractive base tuned\n"
        "\nTests:\n"
        "  pytest -q\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
