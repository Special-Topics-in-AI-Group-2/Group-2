"""seed.py — CSAI415 D2 connectivity smoke-test seeder.

Inserts a small set of sample papers into MongoDB (chunks collection) and
creates the matching Author/Paper/Topic/Venue graph in Neo4j.  Run this
after `docker compose up -d` to verify all three services are reachable
before the team runs the real ingest pipeline.

Usage
-----
    python seed.py                          # all defaults
    python seed.py --mongo-uri mongodb://localhost:27017
    python seed.py --neo4j-password mypass
    python seed.py --wipe                   # drop existing seed data first

What it checks
--------------
  [MongoDB]  connects, inserts 3 documents × 4 chunks = 12 chunk docs,
             then reads them back and prints a count.
  [Neo4j]    connects, creates uniqueness constraints, merges
             Author/Paper/Topic/Venue nodes and relationships,
             then runs a MATCH to count every node label and prints it.

Environment variables (all optional — CLI flags take priority)
--------------------------------------------------------------
    MONGO_URI        default: mongodb://localhost:27017
    MONGO_DB         default: csai415
    NEO4J_URI        default: bolt://localhost:7687
    NEO4J_USERNAME   default: neo4j
    NEO4J_PASSWORD   default: csai415pass
    NEO4J_DATABASE   default: neo4j
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

PAPERS = [
    {
        "paper_id":  "P001",
        "title":     "Attention Is All You Need",
        "authors":   ["Vaswani", "Shazeer", "Parmar", "Uszkoreit"],
        "venue":     "NeurIPS",
        "year":      2017,
        "topics":    ["Transformers", "NLP", "Self-Attention"],
        "doi":       "10.48550/arXiv.1706.03762",
        "chunks": [
            {
                "chunk_id":      "p001_c0",
                "chunk_index":   0,
                "page_start":    1,
                "page_end":      2,
                "text": (
                    "We propose a new simple network architecture, the Transformer, "
                    "based solely on attention mechanisms, dispensing with recurrence "
                    "and convolutions entirely."
                ),
            },
            {
                "chunk_id":      "p001_c1",
                "chunk_index":   1,
                "page_start":    3,
                "page_end":      4,
                "text": (
                    "Multi-head attention allows the model to jointly attend to "
                    "information from different representation subspaces at different "
                    "positions, greatly improving translation quality."
                ),
            },
        ],
    },
    {
        "paper_id":  "P002",
        "title":     "BERT: Pre-training of Deep Bidirectional Transformers",
        "authors":   ["Devlin", "Chang", "Lee", "Toutanova"],
        "venue":     "NAACL",
        "year":      2019,
        "topics":    ["BERT", "NLP", "Pre-training", "Transformers"],
        "doi":       "10.48550/arXiv.1810.04805",
        "chunks": [
            {
                "chunk_id":      "p002_c0",
                "chunk_index":   0,
                "page_start":    1,
                "page_end":      2,
                "text": (
                    "BERT is designed to pre-train deep bidirectional representations "
                    "from unlabelled text by jointly conditioning on both left and "
                    "right context in all layers."
                ),
            },
            {
                "chunk_id":      "p002_c1",
                "chunk_index":   1,
                "page_start":    5,
                "page_end":      6,
                "text": (
                    "Fine-tuning BERT on eleven NLP tasks achieves state-of-the-art "
                    "results, outperforming task-specific architectures by a large margin."
                ),
            },
        ],
    },
    {
        "paper_id":  "P003",
        "title":     "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors":   ["Lewis", "Perez", "Piktus", "Petroni"],
        "venue":     "NeurIPS",
        "year":      2020,
        "topics":    ["RAG", "NLP", "Information Retrieval", "Generation"],
        "doi":       "10.48550/arXiv.2005.11401",
        "chunks": [
            {
                "chunk_id":      "p003_c0",
                "chunk_index":   0,
                "page_start":    1,
                "page_end":      2,
                "text": (
                    "We combine parametric and non-parametric memory for language "
                    "generation, using a dense retriever to fetch relevant documents "
                    "that condition a seq2seq model."
                ),
            },
            {
                "chunk_id":      "p003_c1",
                "chunk_index":   1,
                "page_start":    4,
                "page_end":      5,
                "text": (
                    "RAG models outperform parametric-only seq2seq baselines on "
                    "open-domain QA benchmarks, with retrieved passages providing "
                    "grounding for factual generation."
                ),
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# MongoDB helpers
# ---------------------------------------------------------------------------

def seed_mongodb(uri: str, db_name: str, wipe: bool) -> None:
    print("\n[MongoDB] Connecting …")
    try:
        from pymongo import MongoClient, UpdateOne
    except ImportError:
        sys.exit("  pymongo not found. Run: pip install pymongo")

    client = MongoClient(uri, serverSelectionTimeoutMS=5_000)
    try:
        client.admin.command("ping")
    except Exception as exc:
        sys.exit(f"  Cannot reach MongoDB at {uri}: {exc}")

    db          = client[db_name]
    chunks_col  = db["chunks"]
    docs_col    = db["documents"]

    if wipe:
        chunks_col.delete_many({"chunk_id": {"$in": [
            c["chunk_id"] for p in PAPERS for c in p["chunks"]
        ]}})
        docs_col.delete_many({"paper_id": {"$in": [p["paper_id"] for p in PAPERS]}})
        print("  Wiped existing seed data.")

    now = datetime.now(timezone.utc).isoformat()
    chunk_ops = []
    doc_ops   = []

    for paper in PAPERS:
        # document-level record
        doc = {
            "_id":       paper["paper_id"],
            "paper_id":  paper["paper_id"],
            "title":     paper["title"],
            "authors":   "; ".join(paper["authors"]),
            "venue":     paper["venue"],
            "year":      paper["year"],
            "doi":       paper["doi"],
            "topics":    paper["topics"],
            "ingested_at": now,
        }
        doc_ops.append(UpdateOne({"_id": paper["paper_id"]}, {"$set": doc}, upsert=True))

        # chunk records
        for chunk in paper["chunks"]:
            doc_chunk = {
                "_id":         chunk["chunk_id"],
                "chunk_id":    chunk["chunk_id"],
                "doc_id":      paper["paper_id"],
                "title":       paper["title"],
                "authors":     "; ".join(paper["authors"]),
                "year":        paper["year"],
                "venue":       paper["venue"],
                "doi":         paper["doi"],
                "chunk_index": chunk["chunk_index"],
                "page_start":  chunk["page_start"],
                "page_end":    chunk["page_end"],
                "text":        chunk["text"],
                "token_estimate": len(chunk["text"].split()),
                "provenance": {
                    "filename":   f"{paper['paper_id'].lower()}.pdf",
                    "page_start": chunk["page_start"],
                    "page_end":   chunk["page_end"],
                    "page_range": (
                        f"p. {chunk['page_start']}"
                        if chunk["page_start"] == chunk["page_end"]
                        else f"pp. {chunk['page_start']}-{chunk['page_end']}"
                    ),
                },
                "ingested_at": now,
            }
            chunk_ops.append(
                UpdateOne({"_id": chunk["chunk_id"]}, {"$set": doc_chunk}, upsert=True)
            )

    docs_col.bulk_write(doc_ops, ordered=False)
    result = chunks_col.bulk_write(chunk_ops, ordered=False)

    total_chunks = chunks_col.count_documents({})
    total_docs   = docs_col.count_documents({})

    print(f"  ✓ Connected to {db_name}")
    print(f"  ✓ Upserted {len(PAPERS)} documents into '{db_name}.documents'")
    print(f"  ✓ Upserted {result.upserted_count + result.modified_count} chunks into '{db_name}.chunks'")
    print(f"  ✓ Total documents in collection: {total_docs}")
    print(f"  ✓ Total chunks in collection:    {total_chunks}")

    # Quick read-back spot check
    sample = chunks_col.find_one({"doc_id": "P001"})
    if sample:
        print(f"\n  Sample chunk read-back:")
        print(f"    chunk_id : {sample['chunk_id']}")
        print(f"    title    : {sample['title']}")
        print(f"    pages    : {sample['provenance']['page_range']}")
        print(f"    text     : {sample['text'][:80]}…")

    client.close()


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

CONSTRAINTS = [
    "CREATE CONSTRAINT paper_id_unique   IF NOT EXISTS FOR (p:Paper)  REQUIRE p.paper_id IS UNIQUE",
    "CREATE CONSTRAINT author_name_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.name     IS UNIQUE",
    "CREATE CONSTRAINT topic_name_unique  IF NOT EXISTS FOR (t:Topic)  REQUIRE t.name     IS UNIQUE",
    "CREATE CONSTRAINT venue_name_unique  IF NOT EXISTS FOR (v:Venue)  REQUIRE v.name     IS UNIQUE",
]

UPSERT_CYPHER = """
UNWIND $rows AS row
MERGE (p:Paper {paper_id: row.paper_id})
SET   p.title = row.title,
      p.year  = row.year,
      p.doi   = row.doi

FOREACH (author_name IN row.authors |
    MERGE (a:Author {name: author_name})
    MERGE (a)-[:WROTE]->(p)
)
FOREACH (topic_name IN row.topics |
    MERGE (t:Topic {name: topic_name})
    MERGE (p)-[:ABOUT]->(t)
)
FOREACH (_ IN CASE WHEN row.venue <> '' THEN [1] ELSE [] END |
    MERGE (v:Venue {name: row.venue})
    MERGE (p)-[:PUBLISHED_IN]->(v)
)
"""

COUNT_CYPHER = """
MATCH (p:Paper)  WITH count(p) AS papers
CALL { MATCH (a:Author) RETURN count(a) AS authors }
CALL { MATCH (t:Topic)  RETURN count(t) AS topics  }
CALL { MATCH (v:Venue)  RETURN count(v) AS venues  }
CALL { MATCH ()-[r:WROTE]->()        RETURN count(r) AS wrote }
CALL { MATCH ()-[r:ABOUT]->()        RETURN count(r) AS about }
CALL { MATCH ()-[r:PUBLISHED_IN]->() RETURN count(r) AS pub_in }
RETURN papers, authors, topics, venues, wrote, about, pub_in
"""

SAMPLE_CYPHER = """
MATCH (a:Author)-[:WROTE]->(p:Paper)-[:ABOUT]->(t:Topic)
WHERE p.paper_id = 'P001'
RETURN a.name AS author, p.title AS title, t.name AS topic
LIMIT 5
"""


def seed_neo4j(
    uri: str,
    username: str,
    password: str,
    database: str,
    wipe: bool,
) -> None:
    print("\n[Neo4j] Connecting …")
    try:
        from neo4j import GraphDatabase
        from neo4j.exceptions import AuthError, ServiceUnavailable
    except ImportError:
        sys.exit("  neo4j driver not found. Run: pip install neo4j")

    try:
        driver = GraphDatabase.driver(uri, auth=(username, password))
        driver.verify_connectivity()
    except AuthError:
        sys.exit(f"  Authentication failed — check username/password (got {username!r})")
    except ServiceUnavailable:
        sys.exit(f"  Cannot reach Neo4j at {uri} — is it running?")

    if wipe:
        paper_ids = [p["paper_id"] for p in PAPERS]
        driver.execute_query(
            "UNWIND $ids AS id MATCH (p:Paper {paper_id: id}) DETACH DELETE p",
            ids=paper_ids,
            database_=database,
        )
        print("  Wiped existing seed Paper nodes (and their relationships).")

    # Constraints
    for q in CONSTRAINTS:
        driver.execute_query(q, database_=database)

    # Build rows for the upsert query (matches build_graph.py format)
    rows = [
        {
            "paper_id": p["paper_id"],
            "title":    p["title"],
            "authors":  p["authors"],          # list[str]
            "venue":    p["venue"],
            "year":     p["year"],
            "doi":      p["doi"],
            "topics":   p["topics"],            # list[str]
        }
        for p in PAPERS
    ]

    driver.execute_query(UPSERT_CYPHER, rows=rows, database_=database)

    # Count every node type
    records, _, _ = driver.execute_query(COUNT_CYPHER, database_=database)
    counts = dict(records[0]) if records else {}

    print(f"  ✓ Connected to {uri} (database: {database})")
    print(f"  ✓ Upserted {len(PAPERS)} Paper nodes")
    print()
    print(f"  Node counts:")
    print(f"    Paper   : {counts.get('papers',  0)}")
    print(f"    Author  : {counts.get('authors', 0)}")
    print(f"    Topic   : {counts.get('topics',  0)}")
    print(f"    Venue   : {counts.get('venues',  0)}")
    print()
    print(f"  Relationship counts:")
    print(f"    (Author)-[:WROTE]->(Paper)       : {counts.get('wrote',  0)}")
    print(f"    (Paper)-[:ABOUT]->(Topic)        : {counts.get('about',  0)}")
    print(f"    (Paper)-[:PUBLISHED_IN]->(Venue) : {counts.get('pub_in', 0)}")

    # Quick read-back spot check — two-hop traversal
    rows_back, _, _ = driver.execute_query(SAMPLE_CYPHER, database_=database)
    if rows_back:
        print(f"\n  Sample two-hop read-back (P001 authors → topics):")
        for rec in rows_back:
            print(f"    {rec['author']} wrote '{rec['title']}' → topic: {rec['topic']}")

    driver.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CSAI415 D2 — seed MongoDB and Neo4j with sample paper data.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mongo-uri",
        default=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
        help="MongoDB connection URI.",
    )
    p.add_argument(
        "--mongo-db",
        default=os.getenv("MONGO_DB", "csai415"),
        help="MongoDB database name.",
    )
    p.add_argument(
        "--neo4j-uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j Bolt URI.",
    )
    p.add_argument(
        "--neo4j-username",
        default=os.getenv("NEO4J_USERNAME", "neo4j"),
        help="Neo4j username.",
    )
    p.add_argument(
        "--neo4j-password",
        default=os.getenv("NEO4J_PASSWORD", "csai415pass"),
        help="Neo4j password.",
    )
    p.add_argument(
        "--neo4j-database",
        default=os.getenv("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database name.",
    )
    p.add_argument(
        "--wipe",
        action="store_true",
        help="Delete existing seed records before inserting (safe re-run).",
    )
    p.add_argument(
        "--mongo-only", action="store_true", help="Seed MongoDB only."
    )
    p.add_argument(
        "--neo4j-only", action="store_true", help="Seed Neo4j only."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(textwrap.dedent(f"""
    ╔══════════════════════════════════════════════╗
    ║        CSAI415 D2 — Seed Script              ║
    ╚══════════════════════════════════════════════╝
    Papers   : {len(PAPERS)}
    Chunks   : {sum(len(p['chunks']) for p in PAPERS)}
    Wipe     : {args.wipe}
    """).strip())

    run_mongo = not args.neo4j_only
    run_neo4j = not args.mongo_only

    if run_mongo:
        seed_mongodb(
            uri=args.mongo_uri,
            db_name=args.mongo_db,
            wipe=args.wipe,
        )

    if run_neo4j:
        seed_neo4j(
            uri=args.neo4j_uri,
            username=args.neo4j_username,
            password=args.neo4j_password,
            database=args.neo4j_database,
            wipe=args.wipe,
        )

    print(textwrap.dedent("""
    ──────────────────────────────────────────────
    All checks passed. Services are reachable.

    Next steps:
      • Run the real ingestion:  python ingest.py --pdf_dir ./papers
      • Build the graph:         python build_graph.py --csv papers.csv
      • Start the API:           uvicorn api:app --reload
      • Neo4j browser:           http://localhost:7474
      • Qdrant dashboard:        http://localhost:6333/dashboard
    ──────────────────────────────────────────────
    """).strip())


if __name__ == "__main__":
    main()
