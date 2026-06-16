"""api.py — Basic FastAPI service for CSAI415 Deliverable 2.

Endpoints:
  GET  /health
  GET  /search?query=...&top_k=5
  POST /feedback          — record user helpful/not-helpful signal (feeds D1 AdaptiveAlphaTable)
  POST /ingest            — trigger PDF ingestion pipeline
  GET  /stats             — return run-card metrics and retriever health
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from retriever import HybridRetriever


app = FastAPI(title="PDF Papers AI Agent", version="D2")
retriever: HybridRetriever | None = None

# In-memory feedback log — will be wired to AdaptiveAlphaTable in D3
_feedback_log: list[dict] = []


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
    # FIX #3: optional alpha param so D1's AdaptiveAlphaTable output can be passed directly
    alpha: Optional[float] = Query(None, ge=0.0, le=1.0, description="BM25 weight (D1 alpha convention); overrides bm25_weight/dense_weight when supplied"),
) -> dict:
    if retriever is None:
        raise HTTPException(status_code=500, detail="Retriever is not initialized")

    if alpha is not None:
        # D1 convention: alpha IS the BM25 weight
        bm25_weight = alpha
        dense_weight = 1.0 - alpha
    elif bm25_weight + dense_weight == 0:
        raise HTTPException(status_code=400, detail="At least one retrieval weight must be greater than 0")
    else:
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


# ---------------------------------------------------------------------------
# FIX #6 — Missing endpoints required by the project brief
# ---------------------------------------------------------------------------

class FeedbackRequest(BaseModel):
    """POST /feedback body — records user signal for D1's AdaptiveAlphaTable."""
    query: str
    chunk_id: str
    topic_id: str               # used as key in AdaptiveAlphaTable
    alpha_used: float           # the alpha that produced this result
    helpful: bool               # True = positive signal, False = negative


@app.post("/feedback")
def feedback(body: FeedbackRequest) -> dict:
    """Record a helpful/not-helpful signal for a retrieved chunk.

    Stores the event in an in-memory log (persisted to feedback_log.jsonl).
    In D3 this will be wired to AdaptiveAlphaTable.update(topic_id, alpha_used, helpful).
    """
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "query": body.query,
        "chunk_id": body.chunk_id,
        "topic_id": body.topic_id,
        "alpha_used": body.alpha_used,
        "helpful": body.helpful,
    }
    _feedback_log.append(entry)

    # Append to disk so feedback survives restarts
    log_path = Path("feedback_log.jsonl")
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")

    return {"status": "recorded", "total_feedback_events": len(_feedback_log)}


class IngestRequest(BaseModel):
    """POST /ingest body — triggers the PDF ingestion pipeline."""
    pdf_dir: str = "./papers"
    chunk_size: int = 300
    overlap: int = 50
    dry_run: bool = False


@app.post("/ingest")
def ingest(body: IngestRequest) -> dict:
    """Trigger PDF ingestion and reload the BM25 index.

    Calls ingest.ingest_folder() then retriever.reload_chunks() so the
    running API picks up newly indexed documents without a restart.
    """
    try:
        from ingest import ingest_folder
        summaries = ingest_folder(
            pdf_dir=Path(body.pdf_dir),
            mongo_uri="mongodb://localhost:27017",
            qdrant_host="localhost",
            qdrant_port=6333,
            chunk_size=body.chunk_size,
            overlap=body.overlap,
            dry_run=body.dry_run,
        )
        if retriever and not body.dry_run:
            retriever.reload_chunks()   # rebuild BM25 index over new chunks
        return {
            "status": "ok",
            "pdfs_processed": len(summaries),
            "summaries": summaries,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/stats")
def stats() -> dict:
    """Return retriever health and run-card metrics.

    In D3 this will also return the current per-topic alpha values from
    AdaptiveAlphaTable and the latest prequential accuracy from OnlineTopicClassifier.
    """
    chunk_count = len(retriever.chunks) if retriever else 0
    return {
        "status": "ok",
        "retriever": {
            "chunks_loaded": chunk_count,
            "bm25_ready": retriever.bm25 is not None if retriever else False,
            "embedding_model": "BAAI/bge-small-en-v1.5",
        },
        "feedback": {
            "total_events": len(_feedback_log),
            "log_file": "feedback_log.jsonl",
        },
        # D3 hook: AdaptiveAlphaTable summary goes here
        "online_learning": {
            "alpha_table": None,       # wired in D3
            "prequential_accuracy": None,
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
