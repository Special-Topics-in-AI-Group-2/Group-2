# D4 — SLM Integration Eval (zero-shot vs tuned)

Context source: offline abstract retriever

| Backend | Faithfulness | Answer relevance | Mean ms | p95 ms | Cache hits |
|---|---|---|---|---|---|
| extractive | 0.3204 | 0.1881 | 0.2 | 0.3 | 0 |
| base | 0.0000 | 0.0000 | 0.4 | 0.5 | 15 |
| tuned | 0.0000 | 0.0000 | 980.6 | 1572.0 | 1 |
