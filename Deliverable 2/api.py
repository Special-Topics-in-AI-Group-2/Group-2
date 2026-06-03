"""api.py — Basic FastAPI service for CSAI415 Deliverable 2.

Endpoints:
  GET /health
  GET /search?query=...&top_k=5

The /search endpoint uses retriever.py to combine BM25 search over MongoDB chunks
with dense vector search over Qdrant. Returned results include citations using
paper title and page numbers.
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query

from retriever import HybridRetriever


app = FastAPI(title="PDF Papers AI Agent", version="D2")
retriever: HybridRetriever | None = None


@app.on_event("startup")
def startup_event() -> None:
    global retriever
    retriever = HybridRetriever()


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "retrieval-api",
    }


@app.get("/search")
def search(
    query: str = Query(..., min_length=1, description="User search query"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results to return"),
    bm25_weight: float = Query(0.4, ge=0.0, le=1.0, description="BM25 fusion weight"),
    dense_weight: float = Query(0.6, ge=0.0, le=1.0, description="Dense fusion weight"),
) -> dict:
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever is not initialized")

    if bm25_weight + dense_weight == 0:
        raise HTTPException(status_code=400, detail="At least one retrieval weight must be greater than 0")

    total_weight = bm25_weight + dense_weight
    bm25_weight = bm25_weight / total_weight
    dense_weight = dense_weight / total_weight

    results = retriever.search(
        query=query,
        top_k=top_k,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
    )

    formatted_results = []
    for rank, result in enumerate(results, start=1):
        formatted_results.append(
            {
                "rank": rank,
                "chunk_id": result.get("chunk_id"),
                "doc_id": result.get("doc_id"),
                "paper_title": result.get("title", "Unknown Paper"),
                "authors": result.get("authors", ""),
                "year": result.get("year"),
                "venue": result.get("venue", ""),
                "page_start": result.get("page_start"),
                "page_end": result.get("page_end"),
                "citation": result.get("citation"),
                "text": result.get("text", ""),
                "scores": {
                    "bm25_score": result.get("bm25_score", 0.0),
                    "dense_score": result.get("dense_score", 0.0),
                    "hybrid_score": result.get("hybrid_score", 0.0),
                },
            }
        )

    return {
        "query": query,
        "top_k": top_k,
        "retrieval_method": "BM25 + Dense Vector Hybrid Search",
        "weights": {
            "bm25_weight": bm25_weight,
            "dense_weight": dense_weight,
        },
        "count": len(formatted_results),
        "results": formatted_results,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
