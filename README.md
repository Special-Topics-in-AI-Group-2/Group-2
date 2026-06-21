# PDF-Papers AI Agent — Hybrid Retrieval + GraphRAG + Online Learning + AutoML + SLM Tuning

**CSAI415 Course Project — Group 2.**
An AI agent that answers questions over a corpus of real scientific PDFs with
**grounded, page-ranged citations**, combining hybrid retrieval, a knowledge
graph (GraphRAG), online learning, AutoML, and a PEFT/QLoRA-tuned small language
model.

> **The runnable project lives in the [`Deliverable 4 FINAL (standalone)/`](Deliverable%204) folder** — it is
> self-contained and carries the complete D1, D2, and D3 code, so it is the only
> folder you need to run the whole system. **All commands below are run from
> inside it** (`cd "Deliverable 4"`). The `Deliverable 1/2/3/` folders are the
> original per-stage submissions, kept for reference.

**Group 2** · Abdulla Alshaiba, Essa Alshamsi, Ghaith Alneaimi, Salem Hafez,
Yousef Al Refaie · Repository:
https://github.com/Special-Topics-in-AI-Group-2/Group-2/tree/main

| Layer | Deliverable | What it does |
|---|---|---|
| Streaming + AutoML | **D1** | Optuna-tuned hybrid kNN retriever + River online learner (ADWIN drift) |
| Retrieval + Graph | **D2** | PDF ingest → Mongo + Qdrant; BM25+dense hybrid `/search`; Neo4j knowledge graph |
| GraphRAG + Safety | **D3** | Cypher subgraph → chunk expansion → blend + rerank → cited answer; safety filters; ablation |
| **SLM Tuning** | **D4** | **PEFT/QLoRA** fine-tune a small LM; zero-shot vs tuned **inside** GraphRAG; quantize + cache; final eval |

---

## 1. One-command quickstart (offline, no services, no GPU)

```bash
cd "Deliverable 4"
pip install -r requirements.txt
python scripts/download_corpus.py      # fetch ~102 real arXiv PDFs -> data/pdfs/
python scripts/quickstart.py           # grounded demo answers + D4 eval table
```

`quickstart.py` builds the curated Q/A set, answers sample questions with
grounded page-cited evidence, and prints the D4 quality/latency table — all on a
bare laptop with **no MongoDB / Qdrant / Neo4j / GPU** required.

Run the tests any time (from `Deliverable 4/`):

```bash
pytest -q          # 90+ smoke tests; service-dependent ones skip cleanly
```

### Project walkthrough notebook

[`notebook.ipynb`](notebook.ipynb) (in this repo root) is a member-by-member
tour (each person's D1→D2→D3, then the joint D4 demos) of **every component**,
with explanations, contributor attribution, and **runnable cells with embedded
outputs** (Optuna study, the online-learning drift plot, the safety filter, a
grounded cited answer from the SLM, …). It runs offline — open it and **Run All**.
Regenerate it with `python "Deliverable 4/scripts/build_notebook.py" --out notebook.ipynb`.

> **Windows note:** the code prints Unicode tables/box-drawing. Set
> `PYTHONUTF8=1` before running (PowerShell: `$env:PYTHONUTF8="1"`; cmd:
> `set PYTHONUTF8=1`) to avoid cp1252 console errors.

---

## 2. Architecture / dataflow

```
                         ┌──────────────────── ingest (D2) ────────────────────┐
   data/pdfs/*.pdf ──▶ PyMuPDF parse ──▶ chunk(+overlap) ──▶ bge-small embed ──┐│
        │                                          │                           ││
        │                                  MongoDB (chunks, provenance)  Qdrant (vectors)
   data/papers.csv ──▶ build_graph (D2) ──▶ Neo4j  (Author)-[:WROTE]->(Paper)-[:ABOUT]->(Topic)
                                                          │
                                                          ▼
   query ─▶ D1 OnlineTopicClassifier ─▶ topic ─▶ AdaptiveAlphaTable ─▶ alpha (BM25 weight)
                                                          │
                                                          ▼
        GraphRAG executor (D3):  Cypher subgraph ─▶ expand to chunks
                                 + hybrid BM25/dense top-k (alpha) ─▶ blend ─▶ cross-encoder rerank
                                                          │
                              safety filters (provenance + source pinning + injection block)
                                                          ▼
        Answer generator (D4):  extractive | base(zero-shot) | tuned(PEFT/QLoRA)
                                                          ▼
                       grounded answer + [n] citations with page ranges
```

Everything is connected: D1's learner picks the retrieval `alpha`, D2's stores
feed D3's GraphRAG, and D4's tuned SLM writes the prose while the citation list
stays pinned to the retrieved chunks.

---

## 3. Repository layout

```
Group-2-main/                      (repo root — this README, notebook, chat logs)
├── README.md   notebook.ipynb   chatlogs.md
├── Deliverable 1/  Deliverable 2/  Deliverable 3/    (original per-stage submissions)
└── Deliverable 4/                 ◀── the unified, runnable project (run everything here)
    ├── .env.example  requirements.txt  requirements-slm.txt
    ├── docker-compose.yml  Makefile  pytest.ini
    ├── run_d1.py                      # D1 streaming + AutoML orchestrator
    ├── src/                          # D1 library (package: `src`)
    │   ├── data_utils.py automl_utils.py retriever.py metrics.py
    │   ├── evaluation.py online_learner.py  figures/
    ├── app/                          # D2 + D3 + D4 runtime (flat imports)
    │   ├── config.py                  # unified env/DB config (+ .env loader)
    │   ├── ingest.py retriever.py build_graph.py seed.py api.py eval_search.py   # D2
    │   ├── retriever_bridge.py shared_schema.py
    │   ├── graph_selector.py graphrag_executor.py safety.py safety_filters.py    # D3
    │   ├── ablation.py evaluate_graphrag.py d3_evaluation.py gold_qa.json
    │   ├── slm.py build_qa_dataset.py train_slm.py eval_slm.py                    # D4
    │   └── static/chat.html           # dark chat UI (served at http://localhost:8000/)
    ├── scripts/   data/   tests/   reports/
```

---

## 4. Full stack with the chat UI (Docker)

```bash
cd "Deliverable 4"
cp .env.example .env                       # endpoints + model config (auto-loaded)
docker compose up -d                       # MongoDB + Qdrant + Neo4j

cd app
python seed.py                             # smoke-test all three stores
python ingest.py --pdf_dir ../data/pdfs    # parse -> chunk -> embed -> Mongo + Qdrant
python build_graph.py --csv ../data/papers.csv   # build the Neo4j graph
uvicorn api:app --reload                   # start the agent
```

Then open the **chat interface**:

### 👉 http://localhost:8000/  (dark chat UI — ask questions, pick the answer style, see page-cited sources)

The raw API docs are at **http://localhost:8000/docs**.

### API surface (`app/api.py`)

| Method | Route | Purpose |
|---|---|---|
| GET | `/` | **chat UI** |
| GET | `/search?query=...&top_k=5` | D2 hybrid BM25+dense retrieval |
| POST | `/ask` | **GraphRAG + SLM** answer (D1 routes alpha by topic) |
| POST | `/feedback` | helpful/not-helpful → live D1 `AdaptiveAlphaTable` + drift update |
| POST | `/ingest` | run ingestion + hot-reload index |
| GET | `/stats` | retriever health + live online-learning state |
| GET | `/health` | liveness |

The service **boots even if the stores are down** — affected endpoints return
HTTP 503 with a clear message, so `/` and `/health` always work.

Example call:

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"query":"What is the Transformer based on?","slm_backend":"extractive"}'
```

> **Answer styles:** `extractive` = grounded, no model, instant (the default).
> `tuned` / `base` = the SLM phrases the answer (slow on CPU). On a laptop, keep
> it on **Grounded**. To speed up the model path, set a smaller model + fewer
> tokens in `.env` (`SLM_BASE_MODEL=Qwen/Qwen2.5-0.5B-Instruct`,
> `SLM_MAX_NEW_TOKENS=128`) and restart the server.

---

## 5. Deliverable 4 — SLM fine-tuning (PEFT/QLoRA)

```bash
cd "Deliverable 4"
pip install -r requirements-slm.txt        # torch, transformers, peft, accelerate (+ bitsandbytes on GPU)

cd app
python build_qa_dataset.py                 # curate Q/A from the real corpus -> data/qa/
python train_slm.py                        # PEFT/QLoRA -> artifacts/slm_lora + tuning card
python eval_slm.py --backends extractive base tuned   # final quality/latency table
```

* **QLoRA on GPU**: with a CUDA GPU + `bitsandbytes`, the base model loads in
  4-bit (nf4, double-quant) automatically.
* **CPU smoke**: validate the whole pipeline with a tiny model —
  `python train_slm.py --base-model sshleifer/tiny-gpt2 --epochs 2 --max-len 256`.
* **Three backends** share one interface (`app/slm.py`): `extractive` (grounded,
  no model — the safe default), `base` (zero-shot), `tuned` (LoRA adapter).
* **Quantize + cache**: answers are memoised to `artifacts/slm_cache/`, turning
  repeated queries into ~0 ms cache hits.

See [`Deliverable 4/reports/tuning_card.md`](Deliverable%204/reports/tuning_card.md) and
[`Deliverable 4/reports/D4_Final_Report.md`](Deliverable%204/reports/D4_Final_Report.md).

---

## 6. Deliverables 1–3 entry points

```bash
cd "Deliverable 4"
# D1 — streaming learner + AutoML (writes runs/d1/)
python run_d1.py --trials 50

# D3 — GraphRAG answer / ablation (needs the stores up)
cd app
python graphrag_executor.py "Which papers discuss retrieval-augmented generation?"
python ablation.py --gold gold_qa.json --out-dir ../reports/ablation
python eval_search.py --mongo-gt        # /search Recall@5 + latency table
```

---

## 7. Reproducibility

* **Seeds** fixed throughout (corpus, query stream, Optuna `TPESampler(seed=42)`,
  dataset shuffle, `TruncatedSVD(random_state=42)`).
* **Config** centralised in `app/config.py`; one `.env` drives every service.
* **One-command** paths: `scripts/quickstart.py` (offline) and `make` targets.
* **Pinned** dependency ranges in `requirements*.txt`.

## 8. Ethics & licensing

* Corpus = **open-access arXiv papers**, fetched at run time by
  `scripts/download_corpus.py`; metadata, DOIs, and licenses are recorded in
  `Deliverable 4/data/corpus_metadata.json`. Every answer cites paper title + page range.
* The SLM **base-model weights** keep their own license (e.g. Qwen / Llama
  community licenses); we ship only the trained LoRA **adapter** and the code.
* Safety: prompt-injection blocking, provenance validation, and source pinning
  to the approved corpus (`app/safety.py`, `app/safety_filters.py`).

## 9. Teamwork

| Member | Student ID | Primary ownership |
|---|---|---|
| Abdulla Alshaiba | 21003190 | D1 evaluation/visualisation · D3 ablation study |
| Essa Alshamsi | 22001369 | D1 dense retrieval · D2 ingestion · D3 GraphRAG executor |
| Ghaith Alneaimi | 22001613 | D1 corpus builder · D2 hybrid search + FastAPI · D3 subgraph selection |
| Salem Hafez | 22001171 | D1 AutoML/Optuna · D2 Neo4j graph · D3 evaluation (gold Q/A) |
| Yousef Al Refaie | 22000613 | D1 online learner · D2 Docker + metrics · D3 safety |

Deliverable 4 (SLM tuning + integration + demos) was done jointly by all members.
Per-member, unedited AI chat-history links are in [`chatlogs.md`](chatlogs.md).
