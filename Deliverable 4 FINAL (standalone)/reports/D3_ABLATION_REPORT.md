# Deliverable 3 Ablation Report

## Goal

We added `ablation.py` to compare the same gold question set across three retrieval approaches: vector-only search, graph-guided-only retrieval, and the full hybrid GraphRAG approach.

## Metrics Recorded

The script records faithfulness, answer relevance, latency per question, mean latency, p95 latency, citation rate, and context rate. It writes all results into one comparison table so the three approaches can be compared directly.

## Expected Tradeoffs

Vector-only search is usually the fastest because it only uses the D2 retriever. Its weakness is that it may miss structured graph relationships between papers, authors, and topics.

Graph-guided-only retrieval gives stronger provenance and better graph-based grounding when the Cypher-selected subgraph matches the query. Its weakness is lower recall when the graph match is too narrow.

The full hybrid GraphRAG approach is expected to be slightly slower because it uses both graph expansion and vector retrieval, but it should give the best quality/coverage balance because the vector side back-fills missed chunks and the graph side keeps the answer grounded in paper-topic-author structure with citations.

## Recommendation

For the final D3 system, we recommend the full hybrid GraphRAG approach when quality and citation reliability are more important than speed. Use vector-only as the fallback when latency is the main priority.

## How to Generate the Final Result Table

```bash
python ablation.py --gold gold_qa.json --out-dir ablation_results
```

This produces:

- `ablation_results/ablation_detailed_results.json`
- `ablation_results/ablation_comparison.csv`
- `ablation_results/ablation_comparison.md`
