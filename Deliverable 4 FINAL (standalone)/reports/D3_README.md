# Deliverable 3 Files

This folder contains only the new Deliverable 3 files. It does **not** include or modify the Deliverable 2 files.

## Files

- `graph_selector.py`  
  Selects relevant Papers, Topics, and Authors from Neo4j using Cypher. It also expands selected papers into supporting MongoDB chunks.

- `graphrag_executor.py`  
  Uses the graph selector to create a simple GraphRAG answer with citations/page ranges.

- `safety.py`  
  Adds basic safety mitigation: risky query blocking and source pinning.

- `evaluation.py`  
  Lightweight evaluation runner for latency p95, graph success rate, and citation availability.

- `gold_qa_sample.json`  
  Small sample Q/A file for testing evaluation.

- `requirements_d3.txt`  
  Extra dependencies needed by D3.

## Setup

Place this `deliverable_3` folder beside your existing Deliverable 2 project folder, or copy the files into the root when testing.

Install dependencies:

```bash
pip install -r requirements_d3.txt
```

Make sure MongoDB and Neo4j are running from your D2 Docker Compose.

## Run graph selection only

```bash
python graph_selector.py "transformers attention retrieval"
```

## Run graph selection + MongoDB chunk expansion

```bash
python graph_selector.py "transformers attention retrieval" --with-chunks
```

## Run GraphRAG executor

```bash
python graphrag_executor.py "What papers discuss transformers?"
```

## Run evaluation

```bash
python evaluation.py --gold gold_qa_sample.json --out d3_eval_results.json
```

## Notes

The graph selector includes graceful fallback behavior:

- If Neo4j returns no matches, it returns `fallback_no_graph_matches` instead of crashing.
- If MongoDB has no chunks for the selected papers, it returns an empty chunk list instead of crashing.
- Debug print statements show the selected subgraph: papers, topics, authors, and supporting chunks.

## Safety Filters Evidence Update

Added files:
- `safety_filters.py`: validates chunk provenance, pins sources to the approved PDF corpus folder, and rejects prompt-injected or out-of-scope chunks.
- `test_safety_filters_demo.py`: before/after demo showing unsafe, unverified, and out-of-corpus chunks being blocked.

Run the demo:
```bash
python test_safety_filters_demo.py
```

Expected evidence:
- 4 chunks before filtering
- 1 approved chunk after filtering
- 3 blocked chunks with rejection reasons


## Run D3 Ablation: Vector vs Graph vs Hybrid

`ablation.py` runs the same `gold_qa.json` question set in three modes:

1. `vector` — vector / D2 hybrid retriever only.
2. `graph` — graph-guided supporting chunks only.
3. `hybrid` — full GraphRAG blend of vector + graph evidence.

It records faithfulness, answer relevance, mean latency, p95 latency, citation rate, and context rate in one comparison table.

Run:

```bash
python ablation.py --gold gold_qa.json --out-dir ablation_results
```

If the cross-encoder reranker is slow on your laptop, run:

```bash
python ablation.py --gold gold_qa.json --out-dir ablation_results --no-rerank
```

Outputs:

- `ablation_results/ablation_detailed_results.json`
- `ablation_results/ablation_comparison.csv`
- `ablation_results/ablation_comparison.md`

Metric note: by default, `--metric-backend auto` tries RAGAS first, then falls back to a deterministic lexical scoring method if RAGAS or API credentials are unavailable. For final grading, prefer:

```bash
python ablation.py --metric-backend ragas --gold gold_qa.json --out-dir ablation_results
```
