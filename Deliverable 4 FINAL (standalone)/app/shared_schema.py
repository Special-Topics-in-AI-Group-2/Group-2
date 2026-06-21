"""shared_schema.py — Canonical chunk schema for CSAI415 D2 / D3.

FIX #8: D1 and D2 had incompatible chunk representations:

  D1  src/data_utils.py  Chunk
      chunk_id, paper_id, topic_id, page, text, keywords, semantic_tags

  D2  ingest.py          ChunkRecord
      chunk_id, doc_id, title, authors, year, venue, doi,
      page_start, page_end, chunk_index, text, token_estimate,
      provenance, ingested_at

D3's GraphRAG executor imports ChunkRecord from here (not from ingest.py
directly) so there is one authoritative definition.  The helper
`d1_chunk_to_record()` lets D3 convert any leftover D1 synthetic chunks
into the canonical form without changing D1's code.

Field mapping
-------------
  D1 paper_id   → D2 doc_id          (document identifier)
  D1 page       → D2 page_start = page_end   (D1 has only one page number)
  D1 topic_id   → stored in provenance.topic_id (D2 ChunkRecord has no topic)
  D1 keywords   → not stored in ChunkRecord (used by D1 retriever only)
  D1 semantic_tags → not stored (D1-only concept)

D3 usage
--------
    from shared_schema import ChunkRecord, Provenance, d1_chunk_to_record

    # Real ingested chunks (from MongoDB):
    record: ChunkRecord = ChunkRecord(...)

    # Legacy D1 synthetic chunk:
    from src.data_utils import Chunk as D1Chunk
    record = d1_chunk_to_record(d1_chunk)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


# ---------------------------------------------------------------------------
# Canonical schema (matches D2's ingest.py ChunkRecord exactly)
# ---------------------------------------------------------------------------

@dataclass
class Provenance:
    """Source traceability for a chunk — stored as a nested object in MongoDB."""
    filename: str       # bare filename: "attention_is_all_you_need.pdf"
    source_pdf: str     # absolute path to the PDF on disk
    page_start: int     # first page this chunk covers (1-based)
    page_end: int       # last page this chunk covers (1-based, inclusive)
    page_range: str     # human-readable, e.g. "pp. 3-4" or "p. 5"
    # Optional: populated when converting from D1 synthetic data
    topic_id: str = ""


@dataclass
class ChunkRecord:
    """Canonical chunk representation used by D2 ingest, D2 retriever, and D3 GraphRAG.

    This is the single source of truth for what a chunk looks like across
    all deliverables.  MongoDB stores one document per ChunkRecord; Qdrant
    stores the embedding with chunk_id in the payload.
    """
    chunk_id: str           # 16-char hex or seed ID (e.g. "p001_c0")
    doc_id: str             # document identifier (SHA256[:16] for real PDFs, paper_id for seed)
    title: str              # paper title
    authors: str            # semicolon-separated author string
    year: int | None        # publication year
    venue: str              # journal/conference
    doi: str                # DOI string or empty
    page_start: int         # first page covered (1-based)
    page_end: int           # last page covered (1-based, inclusive)
    chunk_index: int        # 0-based position within the document
    text: str               # chunk text
    token_estimate: int     # word count (proxy for token budget)
    provenance: Provenance
    ingested_at: str        # ISO 8601 UTC timestamp

    def to_mongo_doc(self) -> dict[str, Any]:
        """Serialise to a MongoDB-ready dict (chunk_id becomes _id)."""
        d = asdict(self)
        d["_id"] = self.chunk_id
        return d

    @classmethod
    def from_mongo_doc(cls, doc: dict[str, Any]) -> "ChunkRecord":
        """Deserialise from a MongoDB document dict."""
        prov_raw = doc.get("provenance", {})
        provenance = Provenance(
            filename=prov_raw.get("filename", ""),
            source_pdf=prov_raw.get("source_pdf", ""),
            page_start=prov_raw.get("page_start", 1),
            page_end=prov_raw.get("page_end", 1),
            page_range=prov_raw.get("page_range", ""),
            topic_id=prov_raw.get("topic_id", ""),
        )
        return cls(
            chunk_id=doc.get("chunk_id") or str(doc.get("_id", "")),
            doc_id=doc.get("doc_id", ""),
            title=doc.get("title", ""),
            authors=doc.get("authors", ""),
            year=doc.get("year"),
            venue=doc.get("venue", ""),
            doi=doc.get("doi", ""),
            page_start=doc.get("page_start", 1),
            page_end=doc.get("page_end", 1),
            chunk_index=doc.get("chunk_index", 0),
            text=doc.get("text", ""),
            token_estimate=doc.get("token_estimate", 0),
            provenance=provenance,
            ingested_at=doc.get("ingested_at", datetime.now(timezone.utc).isoformat()),
        )


# ---------------------------------------------------------------------------
# FIX #8 — D1 Chunk → ChunkRecord conversion
# ---------------------------------------------------------------------------

def d1_chunk_to_record(d1_chunk: Any) -> ChunkRecord:
    """Convert a D1 synthetic Chunk into the canonical ChunkRecord.

    D1 Chunk fields:
        chunk_id, paper_id, topic_id, page, text, keywords, semantic_tags

    Mapping:
        paper_id   → doc_id
        page       → page_start AND page_end  (D1 has a single page number)
        topic_id   → provenance.topic_id      (D2 has no top-level topic field)
        keywords   → not stored               (D1 retriever-only concept)
        semantic_tags → not stored
    """
    page = getattr(d1_chunk, "page", 1)
    now = datetime.now(timezone.utc).isoformat()

    provenance = Provenance(
        filename=f"{d1_chunk.paper_id}.pdf",
        source_pdf="",                          # no real PDF for synthetic data
        page_start=page,
        page_end=page,
        page_range=f"p. {page}",
        topic_id=getattr(d1_chunk, "topic_id", ""),
    )

    return ChunkRecord(
        chunk_id=d1_chunk.chunk_id,
        doc_id=d1_chunk.paper_id,              # D1 paper_id ≡ D2 doc_id
        title=f"Synthetic paper {d1_chunk.paper_id}",
        authors="",
        year=None,
        venue="",
        doi="",
        page_start=page,
        page_end=page,
        chunk_index=0,
        text=d1_chunk.text,
        token_estimate=len(d1_chunk.text.split()),
        provenance=provenance,
        ingested_at=now,
    )


# ---------------------------------------------------------------------------
# FIX #10 hook — topic label discovery from Neo4j (called by D3 at startup)
# ---------------------------------------------------------------------------

def fetch_topic_labels_from_neo4j(
    neo4j_uri: str = "bolt://localhost:7687",
    neo4j_user: str = "neo4j",
    neo4j_password: str = "csai415pass",
    neo4j_database: str = "neo4j",
) -> list[str]:
    """Query Neo4j for all Topic node names.

    FIX #10: D1's OnlineTopicClassifier was initialised with synthetic labels
    'topic_0'..'topic_7'.  In D3 it must be initialised with real topic names
    from the knowledge graph so the classifier output matches the Alpha table keys.

    D3 usage:
        from shared_schema import fetch_topic_labels_from_neo4j
        from src.online_learner import OnlineTopicClassifier, AdaptiveAlphaTable

        topic_labels = fetch_topic_labels_from_neo4j()
        clf          = OnlineTopicClassifier(topic_labels)
        alpha_table  = AdaptiveAlphaTable(topic_labels, default_alpha=0.4)

    Returns a sorted list of topic name strings, e.g.:
        ["BERT", "Information Retrieval", "NLP", "Pre-training", "RAG", "Transformers"]
    Falls back to an empty list if Neo4j is unreachable (so the caller can
    decide whether to abort or proceed with a stub).
    """
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("[shared_schema] neo4j driver not found — returning empty topic list.")
        return []

    try:
        # Cap connection/retry windows so a missing Neo4j returns in a few
        # seconds (e.g. on API startup) instead of the driver's ~30s default.
        with GraphDatabase.driver(
            neo4j_uri, auth=(neo4j_user, neo4j_password),
            connection_timeout=3, connection_acquisition_timeout=3,
            max_transaction_retry_time=3,
        ) as driver:
            records, _, _ = driver.execute_query(
                "MATCH (t:Topic) RETURN t.name AS name ORDER BY t.name",
                database_=neo4j_database,
            )
            return [r["name"] for r in records if r["name"]]
    except Exception as exc:
        print(f"[shared_schema] Could not fetch topics from Neo4j: {exc}")
        return []
