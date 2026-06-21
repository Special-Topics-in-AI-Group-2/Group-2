"""build_graph.py — Neo4j graph builder for CSAI415 Deliverable 2.

Creates a knowledge graph from a CSV metadata file with these columns:
    paper id, title, authors, venue, year, topics

Graph schema:
    (:Author)-[:WROTE]->(:Paper)
    (:Paper)-[:ABOUT]->(:Topic)
    (:Paper)-[:PUBLISHED_IN]->(:Venue)

Example:
    python build_graph.py --csv papers.csv

Environment variables:
    NEO4J_URI       e.g. bolt://localhost:7687 or neo4j+s://<aura-host>
    NEO4J_USERNAME  default: neo4j
    NEO4J_PASSWORD  required unless supplied by --password
    NEO4J_DATABASE  default: neo4j

Installation:
    pip install neo4j
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from pathlib import Path
from typing import Any

# Load unified config so NEO4J_* / MONGO_* defaults (and an optional .env) are
# applied — this is what supplies NEO4J_PASSWORD=csai415pass without a CLI flag.
try:
    import config  # noqa: F401  (imported for side effects: populates os.environ)
except Exception:  # noqa: BLE001
    pass

LOGGER = logging.getLogger("build_graph")

EXPECTED_ALIASES = {
    "paper_id": {"paper id", "paper_id", "paperid", "id"},
    "title": {"title", "paper title"},
    "authors": {"authors", "author"},
    "venue": {"venue", "journal", "conference", "published_in"},
    "year": {"year", "publication year", "publication_year"},
    "topics": {"topics", "topic", "keywords"},
}


def normalize_column_name(value: str) -> str:
    """Normalize CSV header names for flexible matching."""
    return " ".join(value.strip().lower().replace("-", " ").replace("_", " ").split())


def resolve_columns(fieldnames: list[str] | None) -> dict[str, str]:
    """Map required logical column names to actual CSV headers."""
    if not fieldnames:
        raise ValueError("The CSV file has no header row.")

    normalized_to_actual = {
        normalize_column_name(field): field for field in fieldnames if field
    }
    resolved: dict[str, str] = {}

    for required, aliases in EXPECTED_ALIASES.items():
        for alias in aliases:
            candidate = normalized_to_actual.get(normalize_column_name(alias))
            if candidate:
                resolved[required] = candidate
                break

    missing = sorted(set(EXPECTED_ALIASES) - set(resolved))
    if missing:
        raise ValueError(
            "CSV is missing required column(s): "
            + ", ".join(missing)
            + f". Found columns: {', '.join(fieldnames)}"
        )
    return resolved


def split_multi_value(value: str, separator: str | None = None) -> list[str]:
    """Split authors or topics while preserving clean unique values.

    A supplied separator takes priority. Without one, the function looks for
    semicolon or pipe separators, then falls back to comma-separated values.
    """
    raw = (value or "").strip()
    if not raw:
        return []

    if separator:
        parts = raw.split(separator)
    elif ";" in raw:
        parts = raw.split(";")
    elif "|" in raw:
        parts = raw.split("|")
    else:
        parts = raw.split(",")

    cleaned: list[str] = []
    seen: set[str] = set()
    for part in parts:
        item = " ".join(part.strip().split())
        key = item.casefold()
        if item and key not in seen:
            cleaned.append(item)
            seen.add(key)
    return cleaned


def parse_year(value: str) -> int | None:
    """Parse a year, returning None for blank or invalid values."""
    raw = (value or "").strip()
    if not raw:
        return None
    try:
        year = int(float(raw))
    except ValueError:
        LOGGER.warning("Invalid year %r; storing as null.", raw)
        return None
    if not 1000 <= year <= 9999:
        LOGGER.warning("Out-of-range year %r; storing as null.", raw)
        return None
    return year


def load_csv_rows(
    csv_path: Path,
    authors_separator: str | None = None,
    topics_separator: str | None = None,
) -> list[dict[str, Any]]:
    """Read and validate the metadata CSV into Neo4j-ready dictionaries."""
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    parsed_rows: list[dict[str, Any]] = []
    skipped = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        columns = resolve_columns(reader.fieldnames)

        for line_number, raw_row in enumerate(reader, start=2):
            paper_id = (raw_row.get(columns["paper_id"]) or "").strip()
            title = (raw_row.get(columns["title"]) or "").strip()

            if not paper_id or not title:
                skipped += 1
                LOGGER.warning(
                    "Skipping CSV line %d because paper id or title is empty.",
                    line_number,
                )
                continue

            parsed_rows.append(
                {
                    "paper_id": paper_id,
                    "title": title,
                    "authors": split_multi_value(
                        raw_row.get(columns["authors"], ""),
                        authors_separator,
                    ),
                    "venue": (raw_row.get(columns["venue"]) or "").strip(),
                    "year": parse_year(raw_row.get(columns["year"], "")),
                    "topics": split_multi_value(
                        raw_row.get(columns["topics"], ""),
                        topics_separator,
                    ),
                }
            )

    if skipped:
        LOGGER.info("Skipped %d invalid CSV row(s).", skipped)
    return parsed_rows


CONSTRAINT_QUERIES = [
    "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
    "CREATE CONSTRAINT author_name_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
    "CREATE CONSTRAINT topic_name_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
    "CREATE CONSTRAINT venue_name_unique IF NOT EXISTS FOR (v:Venue) REQUIRE v.name IS UNIQUE",
]


UPSERT_QUERY = """
UNWIND $rows AS row
MERGE (p:Paper {paper_id: row.paper_id})
SET p.title = row.title,
    p.year = row.year

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


STATS_QUERY = """
MATCH (p:Paper)
WITH count(p) AS papers
CALL {
    MATCH (a:Author) RETURN count(a) AS authors
}
CALL {
    MATCH (t:Topic) RETURN count(t) AS topics
}
CALL {
    MATCH (v:Venue) RETURN count(v) AS venues
}
CALL {
    MATCH (:Author)-[r:WROTE]->(:Paper) RETURN count(r) AS wrote_relationships
}
CALL {
    MATCH (:Paper)-[r:ABOUT]->(:Topic) RETURN count(r) AS about_relationships
}
CALL {
    MATCH (:Paper)-[r:PUBLISHED_IN]->(:Venue) RETURN count(r) AS published_in_relationships
}
RETURN papers,
       authors,
       topics,
       venues,
       wrote_relationships,
       about_relationships,
       published_in_relationships,
       wrote_relationships + about_relationships + published_in_relationships AS relationships
"""


def batched(items: list[dict[str, Any]], batch_size: int):
    """Yield non-empty row batches."""
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def build_graph(
    rows: list[dict[str, Any]],
    uri: str,
    username: str,
    password: str,
    database: str,
    batch_size: int,
) -> dict[str, int]:
    """Create constraints, load CSV rows, and return graph entity counts."""
    from neo4j import GraphDatabase

    with GraphDatabase.driver(uri, auth=(username, password)) as driver:
        driver.verify_connectivity()
        LOGGER.info("Connected to Neo4j database %s.", database)

        for query in CONSTRAINT_QUERIES:
            driver.execute_query(query, database_=database)

        for index, batch in enumerate(batched(rows, batch_size), start=1):
            driver.execute_query(UPSERT_QUERY, rows=batch, database_=database)
            LOGGER.info("Loaded batch %d containing %d paper row(s).", index, len(batch))

        records, _, _ = driver.execute_query(STATS_QUERY, database_=database)
        if not records:
            return {
                "papers": 0,
                "authors": 0,
                "topics": 0,
                "venues": 0,
                "wrote_relationships": 0,
                "about_relationships": 0,
                "published_in_relationships": 0,
                "relationships": 0,
            }
        return dict(records[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the CSAI415 Neo4j paper knowledge graph from CSV metadata.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--csv", required=True, type=Path, help="Path to metadata CSV file.")
    parser.add_argument(
        "--uri",
        default=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j connection URI.",
    )
    parser.add_argument(
        "--username",
        default=os.getenv("NEO4J_USERNAME", "neo4j"),
        help="Neo4j username.",
    )
    parser.add_argument(
        "--password",
        default=os.getenv("NEO4J_PASSWORD"),
        help="Neo4j password. Prefer NEO4J_PASSWORD environment variable.",
    )
    parser.add_argument(
        "--database",
        default=os.getenv("NEO4J_DATABASE", "neo4j"),
        help="Neo4j database name.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=500,
        help="Number of paper rows to upload per transaction batch.",
    )
    parser.add_argument(
        "--authors_separator",
        default=None,
        help="Explicit authors delimiter, e.g. ';'. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--topics_separator",
        default=None,
        help="Explicit topics delimiter, e.g. ';'. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate and preview CSV data without connecting to Neo4j.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.batch_size < 1:
        LOGGER.error("--batch_size must be at least 1.")
        return 2

    try:
        rows = load_csv_rows(
            args.csv,
            authors_separator=args.authors_separator,
            topics_separator=args.topics_separator,
        )
    except (FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2

    if not rows:
        LOGGER.error("No valid paper rows were found in the CSV file.")
        return 2

    LOGGER.info("Validated %d paper row(s) from %s.", len(rows), args.csv)

    if args.dry_run:
        print(f"Dry run complete. Valid paper rows: {len(rows)}")
        print("First parsed row:")
        print(rows[0])
        return 0

    if not args.password:
        LOGGER.error(
            "Neo4j password not supplied. Set NEO4J_PASSWORD or use --password."
        )
        return 2

    try:
        from neo4j.exceptions import AuthError, Neo4jError, ServiceUnavailable
    except ImportError:
        LOGGER.error("neo4j driver not found. Run: pip install neo4j")
        return 2

    try:
        stats = build_graph(
            rows=rows,
            uri=args.uri,
            username=args.username,
            password=args.password,
            database=args.database,
            batch_size=args.batch_size,
        )
    except AuthError:
        LOGGER.error("Authentication failed. Check the Neo4j username and password.")
        return 1
    except ServiceUnavailable:
        LOGGER.error("Neo4j is unreachable. Check the URI and that Neo4j is running.")
        return 1
    except Neo4jError as exc:
        LOGGER.error("Neo4j query failed: %s", exc)
        return 1

    print("\nGraph build complete.")
    print(f"Papers:                  {stats['papers']}")
    print(f"Authors:                 {stats['authors']}")
    print(f"Topics:                  {stats['topics']}")
    print(f"Venues:                  {stats['venues']}")
    print(f"Author-[:WROTE]->Paper:  {stats['wrote_relationships']}")
    print(f"Paper-[:ABOUT]->Topic:   {stats['about_relationships']}")
    print(f"Paper-[:PUBLISHED_IN]->Venue: {stats['published_in_relationships']}")
    print(f"Total relationships:     {stats['relationships']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
