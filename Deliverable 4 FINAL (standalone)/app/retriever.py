"""retriever.py — Hybrid BM25 + dense retrieval for CSAI415 Deliverable 2.

This file reads chunk texts from MongoDB, searches them with BM25, searches
Qdrant with dense embeddings, then combines both scores using weighted fusion.

Expected stores from ingest.py / seed.py:
  - MongoDB database: csai415        (FIX #1: was "papers_ai_agent")
  - MongoDB collection: chunks
  - Qdrant collection: chunks

For D3 GraphRAG: import via retriever_bridge.get_retriever() rather than
instantiating HybridRetriever directly — the bridge handles D1 alpha alignment
and provides the shared RetrieverProtocol interface.

Environment variables can override defaults:
  MONGO_URI, MONGO_DB, MONGO_COLLECTION
  QDRANT_URL, QDRANT_COLLECTION
  EMBEDDING_MODEL
"""

from __future__ import annotations

import os
from typing import Any

from pymongo import MongoClient
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "csai415")           # FIX #1: was "papers_ai_agent"; ingest.py and seed.py both write to "csai415"
MONGO_COLLECTION = os.getenv("MONGO_COLLECTION", "chunks")

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "chunks")

# FIX #4: pin to same model version as ingest.py — was "bge-small-en-v1.5", ingest uses "bge-small-en"
# Using v1.5 everywhere (better model); ingest.py must be updated to match (see ingest.py fix)
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")


class HybridRetriever:
    """Hybrid retriever using BM25 from MongoDB chunks and dense vectors from Qdrant."""

    def __init__(self) -> None:
        self.mongo_client = MongoClient(MONGO_URI)
        self.mongo_collection = self.mongo_client[MONGO_DB][MONGO_COLLECTION]

        self.qdrant_client = QdrantClient(url=QDRANT_URL)
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        self.chunks = list(self.mongo_collection.find({}))
        self.chunk_texts = [chunk.get("text", "") for chunk in self.chunks]
        self.tokenized_corpus = [text.lower().split() for text in self.chunk_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    def reload_chunks(self) -> None:
        """Reload MongoDB chunks and rebuild BM25 index after new ingestion."""
        self.chunks = list(self.mongo_collection.find({}))
        self.chunk_texts = [chunk.get("text", "") for chunk in self.chunks]
        self.tokenized_corpus = [text.lower().split() for text in self.chunk_texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None

    def bm25_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Run lexical BM25 search over chunk text stored in MongoDB."""
        if not self.bm25 or not self.chunks:
            return []

        query_tokens = query.lower().split()
        scores = self.bm25.get_scores(query_tokens)

        ranked_indexes = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        results = []
        for index in ranked_indexes:
            chunk = self.chunks[index]
            results.append(
                {
                    "chunk_id": str(chunk.get("chunk_id") or chunk.get("_id")),
                    "doc_id": chunk.get("doc_id"),
                    "title": chunk.get("title", "Unknown Paper"),
                    "authors": chunk.get("authors", ""),
                    "year": chunk.get("year"),
                    "venue": chunk.get("venue", ""),
                    "page_start": chunk.get("page_start"),
                    "page_end": chunk.get("page_end"),
                    "text": chunk.get("text", ""),
                    "provenance": chunk.get("provenance", {}),
                    "bm25_score": float(scores[index]),
                }
            )

        return results

    def dense_search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Run dense vector search in Qdrant."""
        # FIX #5: bge-small-en uses asymmetric retrieval — corpus chunks are encoded
        # with "Represent this sentence: " (see ingest.py), queries must use
        # "Represent this question: " so embeddings land in the correct subspace.
        query_vector = self.embedding_model.encode(
            f"Represent this question: {query}"
        ).tolist()

        hits = self.qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=top_k,
            with_payload=True,
        )

        results = []
        for hit in hits:
            payload = hit.payload or {}
            chunk_id = str(payload.get("chunk_id") or hit.id)

            mongo_chunk = self.mongo_collection.find_one(
                {"$or": [{"chunk_id": chunk_id}, {"_id": chunk_id}]}
            ) or {}

            merged = {**payload, **mongo_chunk}
            results.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": merged.get("doc_id"),
                    "title": merged.get("title", "Unknown Paper"),
                    "authors": merged.get("authors", ""),
                    "year": merged.get("year"),
                    "venue": merged.get("venue", ""),
                    "page_start": merged.get("page_start"),
                    "page_end": merged.get("page_end"),
                    "text": merged.get("text", ""),
                    "provenance": merged.get("provenance", {}),
                    "dense_score": float(hit.score),
                }
            )

        return results

    @staticmethod
    def _normalize(results: list[dict[str, Any]], score_key: str) -> list[dict[str, Any]]:
        """Min-max normalize scores into a 0-1 range."""
        if not results:
            return results

        scores = [float(result.get(score_key, 0.0)) for result in results]
        min_score = min(scores)
        max_score = max(scores)
        norm_key = f"{score_key}_norm"

        for result in results:
            score = float(result.get(score_key, 0.0))
            if max_score == min_score:
                result[norm_key] = 1.0
            else:
                result[norm_key] = (score - min_score) / (max_score - min_score)

        return results

    @staticmethod
    def _page_citation(title: str, page_start: Any, page_end: Any) -> str:
        """Create readable citation text using paper title and page range."""
        if page_start and page_end and page_start != page_end:
            return f"{title}, pages {page_start}-{page_end}"
        if page_start:
            return f"{title}, page {page_start}"
        return f"{title}, page unknown"

    def search(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.4,
        dense_weight: float = 0.6,
        # FIX #3: accept D1-compatible single alpha so AdaptiveAlphaTable.get_alpha(topic_id)
        # can be piped in directly. alpha=None means use bm25_weight/dense_weight as-is.
        alpha: float | None = None,
    ) -> list[dict[str, Any]]:
        """Return top-k hybrid search results with citation fields.

        Alpha alignment with D1's AdaptiveAlphaTable
        --------------------------------------------
        D1 tracks alpha = BM25 weight per topic (0.0 = pure dense, 1.0 = pure BM25).
        Pass alpha directly to override bm25_weight/dense_weight:
            results = retriever.search(query, alpha=alpha_table.get_alpha(topic_id))
        """
        if alpha is not None:
            # D1 convention: alpha IS the BM25 weight
            bm25_weight = float(alpha)
            dense_weight = 1.0 - bm25_weight
        if top_k <= 0:
            return []

        bm25_results = self.bm25_search(query, top_k=top_k * 3)
        # Dense search degrades gracefully: if Qdrant is unreachable or the
        # client/server versions mismatch, fall back to BM25-only retrieval
        # instead of failing the whole query.
        try:
            dense_results = self.dense_search(query, top_k=top_k * 3)
        except Exception as exc:  # noqa: BLE001
            print(f"[retriever] dense (Qdrant) search unavailable ({exc}); using BM25 only.")
            dense_results = []

        bm25_results = self._normalize(bm25_results, "bm25_score")
        dense_results = self._normalize(dense_results, "dense_score")

        combined: dict[str, dict[str, Any]] = {}

        for result in bm25_results:
            chunk_id = result["chunk_id"]
            combined[chunk_id] = {
                **result,
                "bm25_score_norm": result.get("bm25_score_norm", 0.0),
                "dense_score_norm": 0.0,
                "dense_score": 0.0,
            }

        for result in dense_results:
            chunk_id = result["chunk_id"]
            if chunk_id not in combined:
                combined[chunk_id] = {
                    **result,
                    "bm25_score_norm": 0.0,
                    "bm25_score": 0.0,
                    "dense_score_norm": result.get("dense_score_norm", 0.0),
                }
            else:
                combined[chunk_id]["dense_score"] = result.get("dense_score", 0.0)
                combined[chunk_id]["dense_score_norm"] = result.get("dense_score_norm", 0.0)

        final_results = []
        for result in combined.values():
            hybrid_score = (
                bm25_weight * float(result.get("bm25_score_norm", 0.0))
                + dense_weight * float(result.get("dense_score_norm", 0.0))
            )
            title = result.get("title") or "Unknown Paper"
            page_start = result.get("page_start")
            page_end = result.get("page_end")

            result["hybrid_score"] = float(hybrid_score)
            result["citation"] = self._page_citation(title, page_start, page_end)
            final_results.append(result)

        final_results.sort(key=lambda item: item["hybrid_score"], reverse=True)
        return final_results[:top_k]


if __name__ == "__main__":
    retriever = HybridRetriever()
    for item in retriever.search("machine learning", top_k=5):
        print(item["citation"], item["hybrid_score"])
        print(item.get("text", "")[:250])
        print("-" * 80)
