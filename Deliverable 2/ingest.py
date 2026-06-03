"""ingest.py — CSAI415 D2 PDF ingestion pipeline.

Reads a folder of PDF files using PyMuPDF (fitz), extracts per-page text,
splits into overlapping chunks, generates embeddings with bge-small-en, and
writes to two stores:
  - MongoDB  : "chunks" collection (text + metadata + provenance)
  - Qdrant   : "chunks" collection (embeddings, chunk_id as point ID)

Usage
-----
    python ingest.py --pdf_dir ./papers
    python ingest.py --pdf_dir ./papers --mongo_uri mongodb://localhost:27017
                                        --qdrant_host localhost --qdrant_port 6333
    python ingest.py --pdf_dir ./papers --batch_size 64
    python ingest.py --pdf_dir ./papers --dry_run

MongoDB schema (chunks collection)
-----------------------------------
{
    "_id":         "chunk_id (16-char hex)",
    "chunk_id":    "...",
    "doc_id":      "sha256[:16] of PDF bytes",
    "title":       "Attention Is All You Need",
    "authors":     "Vaswani, A.; Shazeer, N.; ...",
    "year":        2017,
    "venue":       "NeurIPS",
    "doi":         "10.48550/arXiv.1706.03762",
    "page_start":  3,
    "page_end":    4,
    "chunk_index": 7,
    "text":        "...",
    "token_estimate": 298,
    "provenance": {
        "filename":   "attention_is_all_you_need.pdf",
        "source_pdf": "/absolute/path/paper.pdf",
        "page_start": 3,
        "page_end":   4,
        "page_range": "pp. 3-4"
    },
    "ingested_at": "2024-06-03T12:00:00+00:00"
}

Dependencies
------------
    pip install pymupdf pymongo qdrant-client sentence-transformers tqdm
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

try:
    import fitz  # PyMuPDF
except ImportError:
    sys.exit("PyMuPDF not found. Run: pip install pymupdf")

try:
    from pymongo import MongoClient, UpdateOne
    from pymongo.errors import BulkWriteError
except ImportError:
    sys.exit("pymongo not found. Run: pip install pymongo")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("sentence-transformers not found. Run: pip install sentence-transformers")

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        PointStruct,
        VectorParams,
    )
except ImportError:
    sys.exit("qdrant-client not found. Run: pip install qdrant-client")

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class PageRecord:
    """Raw text and metadata for a single PDF page."""
    page_num: int          # 1-based
    text: str
    char_count: int
    is_blank: bool


@dataclass
class Provenance:
    """Source traceability for a chunk — stored as a nested object in MongoDB."""
    filename: str    # bare filename: "attention_is_all_you_need.pdf"
    source_pdf: str  # absolute path to the PDF on disk
    page_start: int  # first page this chunk covers (1-based)
    page_end: int    # last page this chunk covers (1-based, inclusive)
    page_range: str  # human-readable, e.g. "pp. 3-4" or "p. 5"


@dataclass
class ChunkRecord:
    """A text chunk with full provenance — maps 1:1 to a MongoDB chunks document."""
    chunk_id: str        # 16-char hex, used as MongoDB _id and Qdrant point ID
    doc_id: str          # sha256[:16] of the PDF file bytes
    title: str           # from PDF metadata or filename
    authors: str         # semicolon-separated author string
    year: int | None     # publication year from PDF metadata
    venue: str           # journal/conference from PDF subject field
    doi: str             # DOI extracted from PDF metadata or first-page text
    page_start: int      # first page this chunk covers (1-based)
    page_end: int        # last page this chunk covers (1-based, inclusive)
    chunk_index: int     # 0-based position within the document
    text: str            # chunk text (~300 words)
    token_estimate: int  # word count (proxy for token budget)
    provenance: Provenance
    ingested_at: str     # ISO 8601 UTC timestamp


@dataclass
class DocRecord:
    """Document-level metadata stored in MongoDB 'documents' collection."""
    doc_id: str
    source_pdf: str   # kept at top level for easy lookup
    filename: str
    title: str
    authors: str
    year: int | None
    venue: str
    doi: str
    page_count: int
    total_chunks: int
    char_count: int
    ingested_at: str


# ---------------------------------------------------------------------------
# PDF parsing
# ---------------------------------------------------------------------------

def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _chunk_id(doc_id: str, page_num: int, chunk_idx: int) -> str:
    key = f"{doc_id}::{page_num}::{chunk_idx}".encode()
    return hashlib.sha256(key).hexdigest()[:16]


def _extract_year(metadata: dict) -> int | None:
    """Pull year from PDF CreationDate string like 'D:20230415...'."""
    raw = metadata.get("creationDate") or metadata.get("modDate") or ""
    m = re.search(r"D:(\d{4})", raw)
    if m:
        return int(m.group(1))
    return None


def _extract_doi(metadata: dict, first_page_text: str) -> str:
    """Extract DOI from PDF metadata fields or first-page text.

    Tries (in order):
    1. The 'doi' key in PDF metadata (set by some publishers).
    2. Any DOI-like pattern in the keywords or subject fields.
    3. A DOI pattern in the first page of body text (common in arXiv papers).
    Returns an empty string if nothing is found.
    """
    doi_pattern = re.compile(r"10\.\d{4,9}/[^\s\"'<>]+", re.IGNORECASE)

    # 1. Dedicated metadata field
    for key in ("doi", "DOI"):
        val = (metadata.get(key) or "").strip()
        if val:
            return val

    # 2. Keywords / subject fields
    for key in ("keywords", "venue", "subject"):
        val = metadata.get(key) or ""
        m = doi_pattern.search(val)
        if m:
            return m.group(0).rstrip(".,)")

    # 3. First-page text scan (arXiv, ACL, IEEE all print the DOI near the top)
    m = doi_pattern.search(first_page_text)
    if m:
        return m.group(0).rstrip(".,)")

    return ""


def _clean_text(text: str) -> str:
    """Normalise whitespace; remove ligature artifacts."""
    # Replace common ligature substitutions from bad font encodings
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = text.replace("\ufb00", "ff").replace("\ufb03", "ffi").replace("\ufb04", "ffl")
    # Collapse runs of whitespace / blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def extract_pages(pdf_path: Path) -> tuple[dict, list[PageRecord]]:
    """Open a PDF and return (metadata_dict, list[PageRecord]).

    metadata_dict includes a 'doi' key resolved from metadata + first-page text.
    Falls back gracefully on encrypted or damaged files.
    """
    doc = fitz.open(str(pdf_path))

    meta = doc.metadata or {}
    # Grab first-page text early so DOI extraction can scan it
    first_page_text = doc[0].get_text("text") if len(doc) > 0 else ""

    metadata = {
        "title":        (meta.get("title") or "").strip() or pdf_path.stem,
        "authors":      (meta.get("author") or "").strip(),
        "venue":        (meta.get("subject") or "").strip(),
        "keywords":     (meta.get("keywords") or "").strip(),
        "creationDate": (meta.get("creationDate") or ""),
        "modDate":      (meta.get("modDate") or ""),
        "producer":     (meta.get("producer") or ""),
        "doi":          _extract_doi(meta, first_page_text),
    }

    pages: list[PageRecord] = []
    for i, page in enumerate(doc, start=1):
        raw = page.get_text("text")
        cleaned = _clean_text(raw)
        pages.append(PageRecord(
            page_num=i,
            text=cleaned,
            char_count=len(cleaned),
            is_blank=len(cleaned) < 20,
        ))

    doc.close()
    return metadata, pages


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def chunk_pages(
    pages: list[PageRecord],
    chunk_size: int = 300,
    overlap: int = 50,
) -> Generator[tuple[str, int, int], None, None]:
    """Sliding-window word-level chunker across the full document text.

    Yields (chunk_text, page_start, page_end) tuples.

    Parameters
    ----------
    pages       : list of PageRecord (blank pages are skipped)
    chunk_size  : target word count per chunk (default 300)
    overlap     : words shared between consecutive chunks (default 50)
    """
    if not pages:
        return

    # Build a flat word list with per-word page attribution
    words: list[str] = []
    word_pages: list[int] = []   # parallel array: page number for each word

    for p in pages:
        if p.is_blank:
            continue
        page_words = p.text.split()
        words.extend(page_words)
        word_pages.extend([p.page_num] * len(page_words))

    if not words:
        return

    stride = max(1, chunk_size - overlap)
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunk_text = " ".join(chunk_words)
        page_start = word_pages[start]
        page_end = word_pages[end - 1]
        yield chunk_text, page_start, page_end
        if end == len(words):
            break
        start += stride


# ---------------------------------------------------------------------------
# Per-file ingestion
# ---------------------------------------------------------------------------

def ingest_pdf(
    pdf_path: Path,
    chunk_size: int = 300,
    overlap: int = 50,
) -> tuple[DocRecord, list[ChunkRecord]]:
    """Full ingestion for a single PDF.  Returns (DocRecord, [ChunkRecord])."""
    pdf_bytes = pdf_path.read_bytes()
    doc_id = _sha256(pdf_bytes)[:16]
    now = datetime.now(timezone.utc).isoformat()
    abs_path = str(pdf_path.resolve())
    filename = pdf_path.name

    metadata, pages = extract_pages(pdf_path)

    year = _extract_year(metadata)
    title = metadata["title"]
    authors = metadata["authors"]
    venue = metadata["venue"]
    doi = metadata["doi"]

    chunks: list[ChunkRecord] = []
    total_chars = sum(p.char_count for p in pages)

    for idx, (text, page_start, page_end) in enumerate(
        chunk_pages(pages, chunk_size=chunk_size, overlap=overlap)
    ):
        page_range = (
            f"p. {page_start}"
            if page_start == page_end
            else f"pp. {page_start}-{page_end}"
        )
        provenance = Provenance(
            filename=filename,
            source_pdf=abs_path,
            page_start=page_start,
            page_end=page_end,
            page_range=page_range,
        )
        chunks.append(ChunkRecord(
            chunk_id=_chunk_id(doc_id, page_start, idx),
            doc_id=doc_id,
            title=title,
            authors=authors,
            year=year,
            venue=venue,
            doi=doi,
            page_start=page_start,
            page_end=page_end,
            chunk_index=idx,
            text=text,
            token_estimate=len(text.split()),
            provenance=provenance,
            ingested_at=now,
        ))

    doc = DocRecord(
        doc_id=doc_id,
        source_pdf=abs_path,
        filename=filename,
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        page_count=len(pages),
        total_chunks=len(chunks),
        char_count=total_chars,
        ingested_at=now,
    )

    return doc, chunks


# ---------------------------------------------------------------------------
# Embedding model
# ---------------------------------------------------------------------------

EMBED_MODEL_NAME = "BAAI/bge-small-en"
EMBED_DIM = 384  # bge-small-en output dimension


def load_embed_model() -> SentenceTransformer:
    """Load bge-small-en (downloads on first use, cached thereafter)."""
    print(f"[embed] Loading model '{EMBED_MODEL_NAME}' ...")
    model = SentenceTransformer(EMBED_MODEL_NAME)
    print(f"[embed] Model ready. Output dim={EMBED_DIM}")
    return model


def embed_chunks(
    model: SentenceTransformer,
    chunks: list[ChunkRecord],
    batch_size: int = 32,
) -> list[list[float]]:
    """Encode chunk texts with bge-small-en.

    bge-small-en is trained with the instruction prefix
    "Represent this sentence: " for asymmetric retrieval.
    We apply it to passage encoding (corpus side) here; the query side
    should use "Represent this question: " at search time.

    Returns a list of float vectors parallel to `chunks`.
    """
    texts = [f"Represent this sentence: {c.text}" for c in chunks]
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,   # cosine similarity via dot product
        convert_to_numpy=True,
    )
    return [v.tolist() for v in vectors]


# ---------------------------------------------------------------------------
# MongoDB helpers
# ---------------------------------------------------------------------------

MONGO_DB_NAME = "csai415"
CHUNKS_COLLECTION = "chunks"
DOCS_COLLECTION = "documents"


def get_db(mongo_uri: str):
    """Return (db, chunks_col, docs_col) and ensure useful indexes exist."""
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5_000)
    client.admin.command("ping")
    db = client[MONGO_DB_NAME]

    chunks_col = db[CHUNKS_COLLECTION]
    docs_col = db[DOCS_COLLECTION]

    # Indexes — idempotent, safe to call on every run
    chunks_col.create_index("doc_id")
    chunks_col.create_index("title")
    chunks_col.create_index("provenance.filename")
    chunks_col.create_index([("title", 1), ("chunk_index", 1)])

    return db, chunks_col, docs_col


def _chunk_to_mongo_doc(c: ChunkRecord) -> dict:
    """Serialise a ChunkRecord to a MongoDB-ready dict.

    The nested Provenance dataclass is converted to a plain dict so it
    stores as a subdocument rather than a Python object reference.
    """
    d = asdict(c)          # recursively converts nested dataclasses too
    d["_id"] = c.chunk_id  # use chunk_id as the Mongo _id
    return d


def upsert_chunks(chunks_col, chunks: list[ChunkRecord]) -> dict:
    """Bulk-upsert chunks by chunk_id (idempotent re-runs)."""
    if not chunks:
        return {"upserted": 0, "modified": 0}

    ops = [
        UpdateOne(
            {"_id": c.chunk_id},
            {"$set": _chunk_to_mongo_doc(c)},
            upsert=True,
        )
        for c in chunks
    ]
    try:
        result = chunks_col.bulk_write(ops, ordered=False)
        return {
            "upserted": result.upserted_count,
            "modified": result.modified_count,
        }
    except BulkWriteError as bwe:
        print(f"  [mongo] BulkWriteError: {bwe.details.get('nInserted', 0)} inserted, "
              f"{len(bwe.details.get('writeErrors', []))} errors")
        return {"upserted": 0, "modified": 0}


def upsert_doc(docs_col, doc: DocRecord) -> None:
    """Upsert a single document record by doc_id."""
    d = asdict(doc)
    d["_id"] = doc.doc_id
    docs_col.update_one({"_id": doc.doc_id}, {"$set": d}, upsert=True)


# ---------------------------------------------------------------------------
# Qdrant helpers
# ---------------------------------------------------------------------------

QDRANT_COLLECTION = "chunks"


def get_qdrant(host: str, port: int) -> QdrantClient:
    """Connect to Qdrant and create the collection if it doesn't exist."""
    client = QdrantClient(host=host, port=port)

    existing = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION not in existing:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        print(f"[qdrant] Created collection '{QDRANT_COLLECTION}' "
              f"(dim={EMBED_DIM}, cosine)")
    else:
        print(f"[qdrant] Collection '{QDRANT_COLLECTION}' already exists.")

    return client


def upsert_qdrant(
    client: QdrantClient,
    chunks: list[ChunkRecord],
    vectors: list[list[float]],
    batch_size: int = 128,
) -> int:
    """Upsert chunk embeddings into Qdrant in batches.

    The point payload mirrors the MongoDB document (without the text body)
    so that search results are self-contained for citation rendering.

    Returns the total number of points upserted.
    """
    total = 0
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_vectors = vectors[i : i + batch_size]

        points = [
            PointStruct(
                id=c.chunk_id,
                vector=v,
                payload={
                    "chunk_id":      c.chunk_id,
                    "doc_id":        c.doc_id,
                    "title":         c.title,
                    "authors":       c.authors,
                    "year":          c.year,
                    "venue":         c.venue,
                    "doi":           c.doi,
                    "page_start":    c.page_start,
                    "page_end":      c.page_end,
                    "chunk_index":   c.chunk_index,
                    "token_estimate": c.token_estimate,
                    # Full provenance subdoc — enables citation without a Mongo round-trip
                    "provenance":    asdict(c.provenance),
                    "ingested_at":   c.ingested_at,
                },
            )
            for c, v in zip(batch_chunks, batch_vectors)
        ]

        client.upsert(collection_name=QDRANT_COLLECTION, points=points)
        total += len(points)

    return total


# ---------------------------------------------------------------------------
# Folder-level runner
# ---------------------------------------------------------------------------

def ingest_folder(
    pdf_dir: Path,
    mongo_uri: str,
    qdrant_host: str,
    qdrant_port: int,
    chunk_size: int = 300,
    overlap: int = 50,
    batch_size: int = 32,
    dry_run: bool = False,
) -> list[dict]:
    """Ingest all PDFs: parse → chunk → embed → MongoDB + Qdrant.

    Returns a list of per-PDF summary dicts.
    """
    pdf_paths = sorted(pdf_dir.glob("**/*.pdf"))
    if not pdf_paths:
        print(f"[ingest] No PDF files found under {pdf_dir}")
        return []

    # ── Load embedding model (always — validates deps before connecting to DBs)
    embed_model = load_embed_model()

    # ── Connect to stores
    chunks_col = docs_col = None
    qdrant_client = None
    if not dry_run:
        print(f"\n[ingest] Connecting to MongoDB at {mongo_uri} ...")
        try:
            _, chunks_col, docs_col = get_db(mongo_uri)
            print(f"[ingest] MongoDB ready  → {MONGO_DB_NAME}.{CHUNKS_COLLECTION}")
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"[ingest] Cannot connect to MongoDB: {exc}\n"
                     "  Start MongoDB or pass --mongo_uri with the correct URI.")

        print(f"[ingest] Connecting to Qdrant at {qdrant_host}:{qdrant_port} ...")
        try:
            qdrant_client = get_qdrant(qdrant_host, qdrant_port)
            print(f"[ingest] Qdrant ready   → collection '{QDRANT_COLLECTION}'")
        except Exception as exc:  # noqa: BLE001
            sys.exit(f"[ingest] Cannot connect to Qdrant: {exc}\n"
                     "  Start Qdrant or pass --qdrant_host / --qdrant_port.")

    print()
    iterator = tqdm(pdf_paths, unit="pdf") if HAS_TQDM else pdf_paths
    summaries: list[dict] = []
    errors: list[dict] = []

    for pdf_path in iterator:
        label = pdf_path.name[:50]
        if HAS_TQDM:
            iterator.set_description(label)  # type: ignore[union-attr]
        else:
            print(f"[ingest] Processing: {pdf_path.name}")

        t0 = time.perf_counter()

        # 1. Parse PDF → chunks
        try:
            doc, chunks = ingest_pdf(pdf_path, chunk_size=chunk_size, overlap=overlap)
        except Exception as exc:  # noqa: BLE001
            print(f"[ingest] ERROR parsing {pdf_path.name}: {exc}")
            errors.append({"file": pdf_path.name, "error": str(exc)})
            continue

        # 2. Generate embeddings
        try:
            vectors = embed_chunks(embed_model, chunks, batch_size=batch_size)
        except Exception as exc:  # noqa: BLE001
            print(f"[ingest] ERROR embedding {pdf_path.name}: {exc}")
            errors.append({"file": pdf_path.name, "error": str(exc)})
            continue

        mongo_upserted = qdrant_upserted = 0

        # 3. Write to MongoDB + Qdrant
        if not dry_run:
            try:
                upsert_doc(docs_col, doc)
                result = upsert_chunks(chunks_col, chunks)
                mongo_upserted = result["upserted"]
            except Exception as exc:  # noqa: BLE001
                print(f"[ingest] ERROR writing {pdf_path.name} to MongoDB: {exc}")
                errors.append({"file": pdf_path.name, "error": str(exc)})
                continue

            try:
                qdrant_upserted = upsert_qdrant(
                    qdrant_client, chunks, vectors, batch_size=128
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[ingest] ERROR writing {pdf_path.name} to Qdrant: {exc}")
                errors.append({"file": pdf_path.name, "error": str(exc)})
                continue

        summaries.append({
            "file":             pdf_path.name,
            "doc_id":           doc.doc_id,
            "title":            doc.title,
            "doi":              doc.doi or "-",
            "year":             doc.year,
            "pages":            doc.page_count,
            "chunks":           doc.total_chunks,
            "mongo_upserted":   mongo_upserted,
            "qdrant_upserted":  qdrant_upserted,
            "elapsed_s":        round(time.perf_counter() - t0, 3),
            "status":           "ok",
        })

    # ── Final summary ────────────────────────────────────────────────────────
    total_chunks  = sum(s["chunks"]          for s in summaries)
    total_pages   = sum(s["pages"]           for s in summaries)
    total_mongo   = sum(s["mongo_upserted"]  for s in summaries)
    total_qdrant  = sum(s["qdrant_upserted"] for s in summaries)

    print("\n" + "=" * 60)
    print(f"  PDFs processed  : {len(summaries)}")
    print(f"  Pages total     : {total_pages}")
    print(f"  Chunks total    : {total_chunks}")
    if not dry_run:
        print(f"  Mongo upserted  : {total_mongo}  → {MONGO_DB_NAME}.{CHUNKS_COLLECTION}")
        print(f"  Qdrant upserted : {total_qdrant} → collection '{QDRANT_COLLECTION}'")
    if errors:
        print(f"  Errors          : {len(errors)}")
        for e in errors:
            print(f"    - {e['file']}: {e['error']}")
    if dry_run:
        print("  (dry run — nothing written to MongoDB or Qdrant)")
    print("=" * 60)

    return summaries


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "CSAI415 D2 — Ingest PDFs, embed with bge-small-en, "
            "save to MongoDB 'chunks' + Qdrant."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--pdf_dir",
        type=Path,
        default=Path("./papers"),
        help="Folder containing PDF files (searched recursively).",
    )
    # MongoDB
    parser.add_argument(
        "--mongo_uri",
        default="mongodb://localhost:27017",
        help="MongoDB connection URI.",
    )
    # Qdrant
    parser.add_argument(
        "--qdrant_host",
        default="localhost",
        help="Qdrant server hostname.",
    )
    parser.add_argument(
        "--qdrant_port",
        type=int,
        default=6333,
        help="Qdrant server port.",
    )
    # Chunking
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=300,
        help="Target word count per chunk.",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=50,
        help="Word overlap between consecutive chunks.",
    )
    # Embedding
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for sentence-transformers encode().",
    )
    # Misc
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Parse, chunk, and embed but do not write to MongoDB or Qdrant.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ingest_folder(
        pdf_dir=args.pdf_dir,
        mongo_uri=args.mongo_uri,
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )


