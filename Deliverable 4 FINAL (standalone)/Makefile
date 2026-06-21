# PDF-Papers AI Agent — common tasks.
# On Windows without `make`, run the equivalent commands shown, or use:
#     python scripts/quickstart.py
#
# Most app commands run from inside app/ because D2/D3 modules import each other
# by bare name (the original deliverable layout).

PY ?= python

.PHONY: help setup setup-slm corpus qa d1 up down seed ingest graph api \
        ask train-slm eval-slm ablation quickstart test clean

help:
	@echo "Targets:"
	@echo "  setup       install core requirements"
	@echo "  setup-slm   install SLM fine-tuning extras (peft/qlora)"
	@echo "  corpus      download the real arXiv PDF corpus -> data/pdfs"
	@echo "  qa          build the curated SLM Q/A dataset -> data/qa"
	@echo "  quickstart  offline end-to-end demo (no services needed)"
	@echo "  d1          run the D1 streaming + AutoML pipeline"
	@echo "  up / down   start / stop MongoDB + Qdrant + Neo4j (docker compose)"
	@echo "  seed        seed the stores with sample data"
	@echo "  ingest      ingest data/pdfs into MongoDB + Qdrant"
	@echo "  graph       build the Neo4j graph from data/papers.csv"
	@echo "  api         run the FastAPI service (/ask /search /feedback /stats)"
	@echo "  train-slm   PEFT/QLoRA fine-tune the answer SLM"
	@echo "  eval-slm    final eval table: extractive vs base vs tuned"
	@echo "  ablation    D3 vector vs graph vs hybrid ablation"
	@echo "  test        run pytest smoke tests"

setup:
	$(PY) -m pip install -r requirements.txt

setup-slm:
	$(PY) -m pip install -r requirements-slm.txt

corpus:
	$(PY) scripts/download_corpus.py

qa:
	cd app && $(PY) build_qa_dataset.py

quickstart:
	$(PY) scripts/quickstart.py

d1:
	$(PY) run_d1.py

up:
	docker compose up -d

down:
	docker compose down

seed:
	cd app && $(PY) seed.py

ingest:
	cd app && $(PY) ingest.py --pdf_dir ../data/pdfs

graph:
	cd app && $(PY) build_graph.py --csv ../data/papers.csv

api:
	cd app && uvicorn api:app --reload

ask:
	cd app && $(PY) graphrag_executor.py "What is the Transformer architecture based on?"

train-slm:
	cd app && $(PY) train_slm.py

eval-slm:
	cd app && $(PY) eval_slm.py

ablation:
	cd app && $(PY) ablation.py --gold gold_qa.json --out-dir ../reports/ablation --no-rerank

test:
	$(PY) -m pytest -q

clean:
	rm -rf artifacts runs ablation_results app/__pycache__ src/__pycache__ tests/__pycache__ .pytest_cache
