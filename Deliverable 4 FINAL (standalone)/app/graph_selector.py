"""graph_selector.py — D3 GraphRAG subgraph selector + MongoDB chunk expansion.

This file is NEW for Deliverable 3. It does not modify any Deliverable 2 files.

What it does:
1. Takes a user query.
2. Extracts keywords.
3. Uses Cypher queries on Neo4j to select relevant Papers, Topics, and Authors.
4. Gracefully falls back if Neo4j returns no results or errors.
5. Expands selected papers into supporting chunk text from MongoDB.
6. Prints the chosen subgraph for debugging/demo purposes.

Expected D2 Neo4j schema:
    (:Author)-[:WROTE]->(:Paper)
    (:Paper)-[:ABOUT]->(:Topic)
    (:Paper)-[:PUBLISHED_IN]->(:Venue)

Expected D2 MongoDB chunks collection fields:
    chunk_id, doc_id, title, authors, year, venue, page_start, page_end,
    chunk_index, text, provenance

Usage:
    python graph_selector.py "transformers attention retrieval" --with-chunks

Environment variables:
    NEO4J_URI=bolt://localhost:7687
    NEO4J_USERNAME=neo4j
    NEO4J_PASSWORD=csai415pass
    NEO4J_DATABASE=neo4j

    MONGO_URI=mongodb://localhost:27017
    MONGO_DB=pdf_agent
    MONGO_COLLECTION=chunks
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import Neo4jError, ServiceUnavailable, AuthError
except ImportError as exc:
    raise SystemExit("Missing dependency: neo4j. Install it with: pip install neo4j") from exc

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except ImportError as exc:
    raise SystemExit("Missing dependency: pymongo. Install it with: pip install pymongo") from exc

# Load unified config so MONGO_DB / NEO4J_* defaults match the rest of the stack
# (and an optional .env is applied).  Safe no-op if config is unavailable.
try:
    import config  # noqa: F401  (imported for side effects: populates os.environ)
except Exception:  # noqa: BLE001
    pass


STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could", "did",
    "do", "does", "for", "from", "give", "how", "i", "in", "into", "is", "it",
    "me", "of", "on", "or", "our", "paper", "papers", "please", "show", "some",
    "tell", "that", "the", "their", "them", "this", "to", "use", "using", "was",
    "we", "what", "when", "where", "which", "who", "why", "with", "about",
    "related", "relevant", "find", "search", "query", "topic", "topics", "author",
    "authors", "explain", "describe", "based", "according",
}


@dataclass
class GraphPaper:
    paper_id: str
    title: str
    year: int | None
    venue: str
    score: float
    matched_keywords: list[str]
    topics: list[str]
    authors: list[str]


@dataclass
class GraphTopic:
    name: str
    paper_count: int
    score: float
    matched_keywords: list[str]


@dataclass
class GraphAuthor:
    name: str
    paper_count: int
    score: float
    matched_keywords: list[str]


@dataclass
class SupportingChunk:
    chunk_id: str
    doc_id: str
    title: str
    authors: str
    year: int | None
    venue: str
    page_start: int | None
    page_end: int | None
    chunk_index: int | None
    text: str
    provenance: dict[str, Any]


@dataclass
class GraphSelectionResult:
    query: str
    keywords: list[str]
    mode: str
    papers: list[GraphPaper]
    topics: list[GraphTopic]
    authors: list[GraphAuthor]
    chunks: list[SupportingChunk]
    warning: str | None = None

    @property
    def paper_ids(self) -> list[str]:
        ids: list[str] = []
        for paper in self.papers:
            if paper.paper_id:
                ids.append(paper.paper_id)
        return ids

    @property
    def paper_titles(self) -> list[str]:
        return [paper.title for paper in self.papers if paper.title]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GraphSelector:
    """Selects a GraphRAG subgraph and expands selected papers to MongoDB chunks."""

    def __init__(
        self,
        neo4j_uri: str | None = None,
        neo4j_username: str | None = None,
        neo4j_password: str | None = None,
        neo4j_database: str | None = None,
        mongo_uri: str | None = None,
        mongo_db: str | None = None,
        mongo_collection: str | None = None,
        verbose: bool = True,
    ) -> None:
        self.neo4j_uri = neo4j_uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_username = neo4j_username or os.getenv("NEO4J_USERNAME", "neo4j")
        self.neo4j_password = neo4j_password or os.getenv("NEO4J_PASSWORD", "csai415pass")
        self.neo4j_database = neo4j_database or os.getenv("NEO4J_DATABASE", "neo4j")

        self.mongo_uri = mongo_uri or os.getenv("MONGO_URI", "mongodb://localhost:27017")
        self.mongo_db = mongo_db or os.getenv("MONGO_DB", "csai415")  # unified DB name (was "pdf_agent")
        self.mongo_collection_name = mongo_collection or os.getenv("MONGO_COLLECTION", "chunks")
        self.verbose = verbose

        self.driver = None
        self.mongo_client = None
        self.mongo_collection = None

        try:
            # Fail fast when Neo4j is down: cap connection + retry windows so a
            # missing graph degrades to the vector/hybrid fallback in seconds
            # instead of the driver's default ~30s exponential backoff.
            self.driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_username, self.neo4j_password),
                connection_timeout=3,
                connection_acquisition_timeout=3,
                max_transaction_retry_time=3,
            )
        except Exception as exc:
            self._log(f"[GraphSelector] Could not create Neo4j driver: {exc}")

        try:
            self.mongo_client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=3000)
            self.mongo_collection = self.mongo_client[self.mongo_db][self.mongo_collection_name]
        except Exception as exc:
            self._log(f"[GraphSelector] Could not create MongoDB client: {exc}")

    def close(self) -> None:
        if self.driver:
            self.driver.close()
        if self.mongo_client:
            self.mongo_client.close()

    def __enter__(self) -> "GraphSelector":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message)

    @staticmethod
    def extract_keywords(query: str, max_keywords: int = 8) -> list[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}|\d{4}", query.lower())
        keywords: list[str] = []
        seen: set[str] = set()

        for token in tokens:
            token = token.strip("-_")
            if not token or token in STOPWORDS or len(token) < 3:
                continue
            if token not in seen:
                keywords.append(token)
                seen.add(token)
            if len(keywords) >= max_keywords:
                break

        return keywords

    def select(
        self,
        query: str,
        top_papers: int = 10,
        top_topics: int = 8,
        top_authors: int = 8,
        with_chunks: bool = False,
        chunks_per_paper: int = 3,
    ) -> GraphSelectionResult:
        """Select graph nodes, optionally expanding paper nodes into MongoDB chunks."""
        keywords = self.extract_keywords(query)
        if not keywords and query.strip():
            keywords = [query.lower().strip()]

        self._print_query_debug(query, keywords)

        if not self.driver:
            warning = "Neo4j driver is not available. Falling back with empty graph selection."
            self._log(f"[GraphSelector] {warning}")
            return GraphSelectionResult(query, keywords, "fallback_no_neo4j", [], [], [], [], warning)

        papers = self._find_papers(keywords, top_papers)
        topics = self._find_topics(keywords, top_topics)
        authors = self._find_authors(keywords, top_authors)

        if not papers and not topics and not authors:
            warning = "Cypher returned no matching papers, topics, or authors. Use vector/hybrid retrieval fallback."
            self._log(f"[GraphSelector] {warning}")
            return GraphSelectionResult(query, keywords, "fallback_no_graph_matches", [], [], [], [], warning)

        chunks: list[SupportingChunk] = []
        if with_chunks:
            chunks = self.expand_papers_to_chunks(papers, chunks_per_paper=chunks_per_paper)

        result = GraphSelectionResult(
            query=query,
            keywords=keywords,
            mode="graph_selected",
            papers=papers,
            topics=topics,
            authors=authors,
            chunks=chunks,
        )
        self._print_subgraph_debug(result)
        return result

    def select_with_chunks(
        self,
        query: str,
        top_papers: int = 10,
        top_topics: int = 8,
        top_authors: int = 8,
        chunks_per_paper: int = 3,
    ) -> GraphSelectionResult:
        """Convenience wrapper for D3 GraphRAG: graph selection + chunk expansion."""
        return self.select(
            query=query,
            top_papers=top_papers,
            top_topics=top_topics,
            top_authors=top_authors,
            with_chunks=True,
            chunks_per_paper=chunks_per_paper,
        )

    def expand_papers_to_chunks(
        self,
        papers: list[GraphPaper],
        chunks_per_paper: int = 3,
    ) -> list[SupportingChunk]:
        """Expand selected Paper nodes into actual supporting chunk text from MongoDB.

        The D2 graph may use paper_id/doc_id/title differently depending on graph loading.
        This function tries multiple safe lookups:
            1. doc_id in selected paper_id values
            2. title in selected paper titles
            3. case-insensitive title regex fallback
        """
        if not papers:
            self._log("[GraphSelector] No papers supplied for MongoDB chunk expansion.")
            return []

        if self.mongo_collection is None:
            self._log("[GraphSelector] MongoDB collection is not available. Returning no chunks.")
            return []

        paper_ids = [p.paper_id for p in papers if p.paper_id]
        titles = [p.title for p in papers if p.title]
        chunks: list[SupportingChunk] = []

        try:
            # First try exact doc_id match, then exact title match.
            mongo_query: dict[str, Any] = {
                "$or": [
                    {"doc_id": {"$in": paper_ids}},
                    {"title": {"$in": titles}},
                ]
            }
            raw_chunks = list(
                self.mongo_collection.find(mongo_query)
                .sort([("title", 1), ("chunk_index", 1)])
            )

            # If graph title differs by casing/spacing, try regex title fallback.
            if not raw_chunks and titles:
                regex_conditions = [
                    {"title": {"$regex": re.escape(title), "$options": "i"}}
                    for title in titles[:10]
                ]
                raw_chunks = list(
                    self.mongo_collection.find({"$or": regex_conditions})
                    .sort([("title", 1), ("chunk_index", 1)])
                )

            if not raw_chunks:
                self._log("[GraphSelector] Papers were found in Neo4j, but no matching MongoDB chunks were found.")
                return []

            # Keep only the first N chunks per paper title/doc_id to control context size.
            per_paper_count: dict[str, int] = {}
            for chunk in raw_chunks:
                key = str(chunk.get("doc_id") or chunk.get("title") or "unknown")
                count = per_paper_count.get(key, 0)
                if count >= chunks_per_paper:
                    continue
                per_paper_count[key] = count + 1
                chunks.append(self._chunk_from_mongo(chunk))

            self._log(f"[GraphSelector] Expanded selected papers to {len(chunks)} MongoDB supporting chunks.")
            return chunks

        except PyMongoError as exc:
            self._log(f"[GraphSelector] MongoDB query failed: {exc}")
            return []
        except Exception as exc:
            self._log(f"[GraphSelector] Unexpected MongoDB expansion error: {exc}")
            return []

    def _find_papers(self, keywords: list[str], limit: int) -> list[GraphPaper]:
        cypher = """
        MATCH (p:Paper)
        OPTIONAL MATCH (p)-[:ABOUT]->(t:Topic)
        OPTIONAL MATCH (a:Author)-[:WROTE]->(p)
        OPTIONAL MATCH (p)-[:PUBLISHED_IN]->(v:Venue)
        WITH p,
             collect(DISTINCT t.name) AS topics,
             collect(DISTINCT a.name) AS authors,
             coalesce(v.name, '') AS venue
        WITH p, topics, authors, venue,
             [kw IN $keywords WHERE
                toLower(coalesce(p.title, '')) CONTAINS kw OR
                toLower(coalesce(toString(p.year), '')) CONTAINS kw OR
                toLower(venue) CONTAINS kw OR
                any(topic IN topics WHERE toLower(topic) CONTAINS kw) OR
                any(author IN authors WHERE toLower(author) CONTAINS kw)
             ] AS matched_keywords
        WITH p, topics, authors, venue, matched_keywords,
             size(matched_keywords) AS keyword_hits,
             reduce(score = 0.0, kw IN matched_keywords |
                score +
                CASE WHEN toLower(coalesce(p.title, '')) CONTAINS kw THEN 3.0 ELSE 0.0 END +
                CASE WHEN any(topic IN topics WHERE toLower(topic) CONTAINS kw) THEN 2.5 ELSE 0.0 END +
                CASE WHEN any(author IN authors WHERE toLower(author) CONTAINS kw) THEN 2.0 ELSE 0.0 END +
                CASE WHEN toLower(venue) CONTAINS kw THEN 1.0 ELSE 0.0 END +
                CASE WHEN toLower(coalesce(toString(p.year), '')) CONTAINS kw THEN 0.5 ELSE 0.0 END
             ) AS score
        WHERE keyword_hits > 0
        RETURN coalesce(p.paper_id, p.doc_id, p.id, '') AS paper_id,
               p.title AS title,
               p.year AS year,
               venue AS venue,
               score AS score,
               matched_keywords AS matched_keywords,
               topics AS topics,
               authors AS authors
        ORDER BY score DESC, year DESC, title ASC
        LIMIT $limit
        """
        records = self._run(cypher, {"keywords": keywords, "limit": limit})
        return [
            GraphPaper(
                paper_id=str(r.get("paper_id") or ""),
                title=str(r.get("title") or ""),
                year=r.get("year"),
                venue=str(r.get("venue") or ""),
                score=float(r.get("score") or 0.0),
                matched_keywords=list(r.get("matched_keywords") or []),
                topics=list(r.get("topics") or []),
                authors=list(r.get("authors") or []),
            )
            for r in records
        ]

    def _find_topics(self, keywords: list[str], limit: int) -> list[GraphTopic]:
        cypher = """
        MATCH (t:Topic)<-[:ABOUT]-(p:Paper)
        WITH t, count(DISTINCT p) AS paper_count,
             [kw IN $keywords WHERE toLower(t.name) CONTAINS kw] AS matched_keywords
        WHERE size(matched_keywords) > 0
        WITH t, paper_count, matched_keywords,
             (size(matched_keywords) * 2.5) + log10(paper_count + 1) AS score
        RETURN t.name AS name,
               paper_count AS paper_count,
               score AS score,
               matched_keywords AS matched_keywords
        ORDER BY score DESC, paper_count DESC, name ASC
        LIMIT $limit
        """
        records = self._run(cypher, {"keywords": keywords, "limit": limit})
        return [
            GraphTopic(
                name=str(r.get("name") or ""),
                paper_count=int(r.get("paper_count") or 0),
                score=float(r.get("score") or 0.0),
                matched_keywords=list(r.get("matched_keywords") or []),
            )
            for r in records
        ]

    def _find_authors(self, keywords: list[str], limit: int) -> list[GraphAuthor]:
        cypher = """
        MATCH (a:Author)-[:WROTE]->(p:Paper)
        WITH a, count(DISTINCT p) AS paper_count,
             [kw IN $keywords WHERE toLower(a.name) CONTAINS kw] AS matched_keywords
        WHERE size(matched_keywords) > 0
        WITH a, paper_count, matched_keywords,
             (size(matched_keywords) * 2.0) + log10(paper_count + 1) AS score
        RETURN a.name AS name,
               paper_count AS paper_count,
               score AS score,
               matched_keywords AS matched_keywords
        ORDER BY score DESC, paper_count DESC, name ASC
        LIMIT $limit
        """
        records = self._run(cypher, {"keywords": keywords, "limit": limit})
        return [
            GraphAuthor(
                name=str(r.get("name") or ""),
                paper_count=int(r.get("paper_count") or 0),
                score=float(r.get("score") or 0.0),
                matched_keywords=list(r.get("matched_keywords") or []),
            )
            for r in records
        ]

    def _run(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.driver:
            return []
        try:
            records, _, _ = self.driver.execute_query(
                cypher,
                params,
                database_=self.neo4j_database,
            )
            return [dict(record) for record in records]
        except (ServiceUnavailable, AuthError) as exc:
            self._log(f"[GraphSelector] Neo4j connection/authentication error: {exc}")
            return []
        except Neo4jError as exc:
            self._log(f"[GraphSelector] Neo4j Cypher query failed: {exc}")
            return []
        except Exception as exc:
            self._log(f"[GraphSelector] Unexpected Neo4j error: {exc}")
            return []

    @staticmethod
    def _chunk_from_mongo(chunk: dict[str, Any]) -> SupportingChunk:
        return SupportingChunk(
            chunk_id=str(chunk.get("chunk_id") or chunk.get("_id") or ""),
            doc_id=str(chunk.get("doc_id") or ""),
            title=str(chunk.get("title") or ""),
            authors=str(chunk.get("authors") or ""),
            year=chunk.get("year"),
            venue=str(chunk.get("venue") or ""),
            page_start=chunk.get("page_start"),
            page_end=chunk.get("page_end"),
            chunk_index=chunk.get("chunk_index"),
            text=str(chunk.get("text") or ""),
            provenance=dict(chunk.get("provenance") or {}),
        )

    def _print_query_debug(self, query: str, keywords: list[str]) -> None:
        self._log("\n========== GRAPH SELECTOR QUERY ==========")
        self._log(query)
        self._log("\n========== EXTRACTED KEYWORDS ==========")
        self._log(", ".join(keywords) if keywords else "No keywords extracted")

    def _print_subgraph_debug(self, result: GraphSelectionResult) -> None:
        self._log("\n========== SELECTED SUBGRAPH ==========")
        self._log(f"Mode: {result.mode}")

        self._log("\nPapers:")
        if not result.papers:
            self._log("  No papers selected")
        for paper in result.papers[:10]:
            self._log(f"  - {paper.title} | score={paper.score:.2f} | matched={paper.matched_keywords}")

        self._log("\nTopics:")
        if not result.topics:
            self._log("  No topics selected")
        for topic in result.topics[:8]:
            self._log(f"  - {topic.name} | papers={topic.paper_count} | score={topic.score:.2f}")

        self._log("\nAuthors:")
        if not result.authors:
            self._log("  No authors selected")
        for author in result.authors[:8]:
            self._log(f"  - {author.name} | papers={author.paper_count} | score={author.score:.2f}")

        if result.chunks:
            self._log("\nSupporting chunks:")
            for chunk in result.chunks[:10]:
                page_range = chunk.provenance.get("page_range") or f"pp. {chunk.page_start}-{chunk.page_end}"
                self._log(f"  - {chunk.title} | chunk={chunk.chunk_id} | {page_range}")


def main() -> None:
    parser = argparse.ArgumentParser(description="D3 GraphRAG graph selector with MongoDB chunk expansion.")
    parser.add_argument("query", help="User query, e.g. 'transformers for retrieval'")
    parser.add_argument("--top-papers", type=int, default=10)
    parser.add_argument("--top-topics", type=int, default=8)
    parser.add_argument("--top-authors", type=int, default=8)
    parser.add_argument("--with-chunks", action="store_true", help="Expand selected papers into MongoDB chunk text")
    parser.add_argument("--chunks-per-paper", type=int, default=3)
    parser.add_argument("--quiet", action="store_true", help="Disable debug print statements")
    args = parser.parse_args()

    with GraphSelector(verbose=not args.quiet) as selector:
        result = selector.select(
            query=args.query,
            top_papers=args.top_papers,
            top_topics=args.top_topics,
            top_authors=args.top_authors,
            with_chunks=args.with_chunks,
            chunks_per_paper=args.chunks_per_paper,
        )

    print("\n========== JSON RESULT ==========")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
