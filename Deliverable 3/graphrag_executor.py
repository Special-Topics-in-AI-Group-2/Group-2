"""graphrag_executor.py — D3 GraphRAG executor with hybrid blend.

This version implements the full D3 GraphRAG pipeline the brief asks for:

    (1) choose a subgraph by Cypher          (graph_selector.GraphSelector)
    (2) expand it to supporting chunks       (graph_selector -> MongoDB)
    (3) blend those chunks with the normal    (retriever_bridge -> D2 HybridRetriever)
        hybrid (BM25 + dense) top-k
    (4) rerank the blended candidate pool by    (cross-encoder, optional)
        query relevance, then keep the top-k
    (5) answer with citations and page ranges

Why blend?
----------
Graph selection is high-precision but recall-limited: it only surfaces chunks
from papers whose title/topic/author literally matched a query keyword.  The D2
hybrid retriever has the opposite profile — broad lexical+semantic recall but no
graph reasoning.  Blending the two:

  * keeps the graph-selected chunks (good provenance, topically anchored), and
  * back-fills with vector/BM25 hits the graph missed, and
  * up-weights chunks that BOTH signals agree on (co-occurrence bonus).

The blend itself (`blend_results`) is a pure function so it can be unit-tested
without MongoDB / Neo4j / Qdrant running.

Ablation support
----------------
`mode` selects which signal(s) feed the final ranking:
    "vector"  -> D2 hybrid retriever only          (vector-only baseline)
    "graph"   -> graph-selected chunks only         (graph-guided)
    "hybrid"  -> blend of both                       (default, full pipeline)

That gives you the three-way ablation table D3 asks for directly from one class.

Usage
-----
    python graphrag_executor.py "What papers discuss transformers?"
    python graphrag_executor.py "attention for retrieval" --mode vector
    python graphrag_executor.py "attention for retrieval" --mode graph
    python graphrag_executor.py "attention for retrieval" --alpha 0.3 --top-k 8
    python graphrag_executor.py "attention for retrieval" --no-rerank
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from typing import Any

from graph_selector import GraphSelector, GraphPaper, SupportingChunk
from safety import filter_safe_chunks, is_risky_query


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class GraphRAGAnswer:
    query: str
    mode: str                       # pipeline mode actually used (see below)
    answer: str
    citations: list[dict[str, Any]]
    blended: list[dict[str, Any]] = field(default_factory=list)   # full ranked, blended evidence
    selected_papers: list[str] = field(default_factory=list)
    selected_topics: list[str] = field(default_factory=list)
    selected_authors: list[str] = field(default_factory=list)
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Score normalisation + blending — PURE FUNCTIONS (no DB needed)
# ---------------------------------------------------------------------------

def _min_max_norm(scores: list[float]) -> list[float]:
    """Min-max normalise a list of scores into [0, 1].

    Mirrors the normalisation D2's HybridRetriever uses internally so the two
    score families are combined on the same footing.  All-equal (or empty)
    inputs map to 1.0 to avoid divide-by-zero collapsing everything to 0.
    """
    if not scores:
        return []
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


def _page_range_str(page_start: Any, page_end: Any) -> str:
    if page_start and page_end and page_start != page_end:
        return f"pp. {page_start}-{page_end}"
    if page_start:
        return f"p. {page_start}"
    return "page unknown"


def _graph_chunk_to_dict(chunk: SupportingChunk, graph_raw_score: float) -> dict[str, Any]:
    """Normalise a SupportingChunk (graph side) into the shared result dict shape."""
    page_start = chunk.page_start
    page_end = chunk.page_end
    page_range = (
        (chunk.provenance or {}).get("page_range")
        or _page_range_str(page_start, page_end)
    )
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "authors": chunk.authors,
        "year": chunk.year,
        "venue": chunk.venue,
        "page_start": page_start,
        "page_end": page_end,
        "page_range": page_range,
        "text": chunk.text,
        "provenance": chunk.provenance or {},
        "hybrid_score": 0.0,       # vector side score (0 = not retrieved by vectors)
        "graph_raw_score": graph_raw_score,
        "in_graph": True,
        "in_vector": False,
    }


def _vector_result_to_dict(res: dict[str, Any]) -> dict[str, Any]:
    """Normalise a D2 retriever result dict into the shared result dict shape."""
    page_start = res.get("page_start")
    page_end = res.get("page_end")
    prov = res.get("provenance") or {}
    page_range = prov.get("page_range") or _page_range_str(page_start, page_end)
    return {
        "chunk_id": str(res.get("chunk_id") or ""),
        "doc_id": res.get("doc_id"),
        "title": res.get("title", "Unknown Paper"),
        "authors": res.get("authors", ""),
        "year": res.get("year"),
        "venue": res.get("venue", ""),
        "page_start": page_start,
        "page_end": page_end,
        "page_range": page_range,
        "text": res.get("text", ""),
        "provenance": prov,
        "hybrid_score": float(res.get("hybrid_score", 0.0) or 0.0),
        "graph_raw_score": 0.0,
        "in_graph": False,
        "in_vector": True,
    }


def _citation_str(title: str, page_start: Any, page_end: Any) -> str:
    title = title or "Unknown Paper"
    if page_start and page_end and page_start != page_end:
        return f"{title}, pages {page_start}-{page_end}"
    if page_start:
        return f"{title}, page {page_start}"
    return f"{title}, page unknown"


def blend_results(
    vector_results: list[dict[str, Any]],
    graph_chunks: list[SupportingChunk],
    paper_scores: dict[str, float] | None = None,
    vector_weight: float = 0.6,
    graph_weight: float = 0.4,
    presence_bonus: float = 0.1,
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Blend graph-selected chunks with D2 hybrid (BM25+dense) results.

    Both score families are min-max normalised, then combined as::

        final = vector_weight * hybrid_norm
              + graph_weight  * graph_norm
              + presence_bonus  (only if the chunk appears in BOTH signals)

    Join key is ``chunk_id``.
    """
    paper_scores = paper_scores or {}

    graph_dicts: list[dict[str, Any]] = []
    for ch in graph_chunks:
        raw = (
            paper_scores.get(ch.doc_id)
            or paper_scores.get(ch.title)
            or 1.0
        )
        graph_dicts.append(_graph_chunk_to_dict(ch, float(raw)))

    vector_dicts = [_vector_result_to_dict(r) for r in vector_results]

    g_norms = _min_max_norm([d["graph_raw_score"] for d in graph_dicts])
    for d, n in zip(graph_dicts, g_norms):
        d["graph_score_norm"] = n
    v_norms = _min_max_norm([d["hybrid_score"] for d in vector_dicts])
    for d, n in zip(vector_dicts, v_norms):
        d["hybrid_score_norm"] = n

    merged: dict[str, dict[str, Any]] = {}

    for d in graph_dicts:
        merged[d["chunk_id"]] = {
            **d,
            "hybrid_score_norm": 0.0,
            "graph_score_norm": d.get("graph_score_norm", 0.0),
        }

    for d in vector_dicts:
        cid = d["chunk_id"]
        if cid in merged:
            merged[cid]["in_vector"] = True
            merged[cid]["hybrid_score"] = d["hybrid_score"]
            merged[cid]["hybrid_score_norm"] = d.get("hybrid_score_norm", 0.0)
            if not merged[cid].get("text"):
                merged[cid]["text"] = d["text"]
            if not merged[cid].get("provenance"):
                merged[cid]["provenance"] = d["provenance"]
        else:
            merged[cid] = {
                **d,
                "graph_score_norm": 0.0,
                "hybrid_score_norm": d.get("hybrid_score_norm", 0.0),
            }

    out: list[dict[str, Any]] = []
    for d in merged.values():
        both = d.get("in_graph") and d.get("in_vector")
        score = (
            vector_weight * float(d.get("hybrid_score_norm", 0.0))
            + graph_weight * float(d.get("graph_score_norm", 0.0))
            + (presence_bonus if both else 0.0)
        )
        d["blend_score"] = float(score)
        d["citation"] = _citation_str(d["title"], d["page_start"], d["page_end"])
        out.append(d)

    out.sort(key=lambda x: x["blend_score"], reverse=True)
    return out[:top_k]




# ---------------------------------------------------------------------------
# Reranking — reorder the blended candidate pool by query relevance
# ---------------------------------------------------------------------------

# Lightweight, CPU-friendly cross-encoder reranker.  Scores each (query, chunk)
# pair jointly, which is more accurate than the bi-encoder cosine used for the
# first-stage dense retrieval.  Loaded lazily and optional — if
# sentence-transformers is unavailable the pipeline keeps the blend order.
DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def build_cross_encoder_scorer(model_name: str = DEFAULT_RERANK_MODEL):
    """Return a score_fn(query, texts)->list[float] backed by a cross-encoder.

    Returns None if sentence-transformers (or the model) cannot be loaded, so
    callers can fall back to the blend ordering without crashing.
    """
    try:
        from sentence_transformers import CrossEncoder
    except Exception:  # noqa: BLE001
        return None
    try:
        model = CrossEncoder(model_name)
    except Exception:  # noqa: BLE001
        return None

    def score_fn(query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [[query, t or ""] for t in texts]
        scores = model.predict(pairs)
        return [float(s) for s in scores]

    return score_fn


def rerank_results(
    query: str,
    results: list[dict[str, Any]],
    score_fn=None,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    """Reorder blended results by query relevance, then keep top_k.

    Pure with respect to the model: the actual relevance model is injected as
    ``score_fn`` (a callable taking the query and a list of chunk texts and
    returning one float per text).  This keeps the function unit-testable
    without loading a cross-encoder.

    If ``score_fn`` is None (no reranker available) the input order — i.e. the
    blend ranking — is preserved, and only the top_k cut is applied.  A
    ``rerank_score`` field is attached to each result when reranking runs.
    """
    if not results:
        return []

    if score_fn is None:
        ordered = list(results)
    else:
        texts = [r.get("text", "") for r in results]
        try:
            scores = score_fn(query, texts)
        except Exception:  # noqa: BLE001
            scores = None
        if not scores or len(scores) != len(results):
            ordered = list(results)
        else:
            for r, s in zip(results, scores):
                r["rerank_score"] = float(s)
            ordered = sorted(
                results, key=lambda r: r.get("rerank_score", 0.0), reverse=True
            )

    if top_k is not None:
        ordered = ordered[:top_k]
    return ordered


# ---------------------------------------------------------------------------
# Answer rendering — build a cited answer string from the final top chunks
# ---------------------------------------------------------------------------

def _page_range_of(chunk: dict[str, Any]) -> str:
    """Best-effort page range string for a result dict.

    Prefers an explicit ``page_range`` (from provenance / blend), then falls
    back to page_start/page_end, then to 'page unknown'.
    """
    pr = chunk.get("page_range")
    if pr:
        return pr
    prov = chunk.get("provenance") or {}
    if prov.get("page_range"):
        return prov["page_range"]
    return _page_range_str(chunk.get("page_start"), chunk.get("page_end"))


def build_answer_with_citations(
    query: str,
    chunks: list[dict[str, Any]],
    max_snippets: int = 5,
    snippet_chars: int = 300,
) -> tuple[str, list[dict[str, Any]]]:
    """Build a grounded answer string with inline, numbered citations.

    Each supporting chunk becomes one cited evidence line carrying a ``[n]``
    marker, and a trailing **Sources** block maps every marker to its paper
    title and page range (plus the source PDF when provenance has it).  This is
    the citation/page-range surface the D3 rubric asks the GraphRAG answer to
    expose.

    The function is deliberately model-free: it stitches the retrieved evidence
    into a citeable answer without inventing content, so every sentence is
    traceable to a (title, page range) pair.  In D4 the prose around the
    markers can be replaced by the PEFT/QLoRA SLM while this same citation
    list is reused unchanged.

    Returns
    -------
    (answer_text, references)
        answer_text : str
            Human-readable answer with ``[n]`` markers and a Sources section.
        references : list[dict]
            One entry per marker: {marker, title, page_range, chunk_id,
            doc_id, source_pdf}.
    """
    if not chunks:
        return (
            "No supporting evidence with citable page ranges was found, so no "
            "grounded answer can be produced.",
            [],
        )

    used = chunks[:max_snippets]

    # Build the reference table first so markers are stable.
    references: list[dict[str, Any]] = []
    for i, ch in enumerate(used, start=1):
        prov = ch.get("provenance") or {}
        references.append(
            {
                "marker": i,
                "title": ch.get("title") or "Unknown Paper",
                "page_range": _page_range_of(ch),
                "chunk_id": ch.get("chunk_id"),
                "doc_id": ch.get("doc_id"),
                "source_pdf": prov.get("source_pdf"),
            }
        )

    # Evidence lines — each grounded snippet carries its [n] marker.
    lines: list[str] = [
        f"Answer to: {query}",
        "",
        "Based on the retrieved sources:",
        "",
    ]
    for ref, ch in zip(references, used):
        snippet = " ".join((ch.get("text") or "").split())[:snippet_chars].rstrip()
        if not snippet:
            # No body text (rare — usually filtered earlier); still cite it
            # in Sources, but don't emit an empty evidence line here.
            continue
        if not snippet.endswith((".", "!", "?")):
            snippet += "..."
        lines.append(
            f"- {snippet} "
            f"[{ref['marker']}] ({ref['title']}, {ref['page_range']})"
        )

    # Sources block — marker -> title + page range (+ source pdf if known).
    lines.append("")
    lines.append("Sources:")
    for ref in references:
        src = f" — {ref['source_pdf']}" if ref.get("source_pdf") else ""
        lines.append(
            f"  [{ref['marker']}] {ref['title']}, {ref['page_range']}{src}"
        )

    return "\n".join(lines), references


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class GraphRAGExecutor:
    """GraphRAG executor that blends graph subgraph chunks with D2 hybrid search.

    The D2 retriever is loaded lazily via `retriever_bridge.get_retriever()` so
    that graph-only runs (and unit tests of `blend_results`) don't require Qdrant.
    """

    def __init__(
        self,
        verbose: bool = True,
        default_alpha: float = 0.4,
        vector_weight: float = 0.6,
        graph_weight: float = 0.4,
        presence_bonus: float = 0.1,
        rerank: bool = True,
        rerank_model: str = DEFAULT_RERANK_MODEL,
        rerank_pool: int | None = None,
    ) -> None:
        self.verbose = verbose
        self.default_alpha = default_alpha
        self.vector_weight = vector_weight
        self.graph_weight = graph_weight
        self.presence_bonus = presence_bonus
        self.rerank = rerank
        self.rerank_model = rerank_model
        self.rerank_pool = rerank_pool      # candidate pool size before rerank (default: top_k*4)

        self.selector = GraphSelector(verbose=verbose)
        self._retriever = None          # lazy — only built when vectors are needed
        self._rerank_scorer = None      # lazy cross-encoder scorer
        self._rerank_loaded = False

    def close(self) -> None:
        self.selector.close()

    def __enter__(self) -> "GraphRAGExecutor":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _get_retriever(self):
        """Lazily construct the D2 hybrid retriever (Mongo + Qdrant + BM25)."""
        if self._retriever is None:
            try:
                from retriever_bridge import get_retriever
                self._retriever = get_retriever(
                    mode="production", default_alpha=self.default_alpha
                )
            except Exception as exc:  # noqa: BLE001
                self._log(f"[GraphRAGExecutor] Could not load D2 retriever: {exc}")
                self._retriever = None
        return self._retriever

    def _get_rerank_scorer(self):
        """Lazily build the cross-encoder scorer; cache None if unavailable."""
        if not self.rerank:
            return None
        if not self._rerank_loaded:
            self._rerank_loaded = True
            self._rerank_scorer = build_cross_encoder_scorer(self.rerank_model)
            if self._rerank_scorer is None:
                self._log(
                    "[GraphRAGExecutor] Cross-encoder reranker unavailable "
                    "(sentence-transformers/model not loaded); keeping blend order."
                )
            else:
                self._log(f"[GraphRAGExecutor] Reranker ready: {self.rerank_model}")
        return self._rerank_scorer

    def _vector_search(self, query: str, top_k: int, alpha: float | None) -> list[dict[str, Any]]:
        retriever = self._get_retriever()
        if retriever is None:
            return []
        try:
            return retriever.search(query=query, top_k=top_k, alpha=alpha)
        except Exception as exc:  # noqa: BLE001
            self._log(f"[GraphRAGExecutor] Vector search failed: {exc}")
            return []

    @staticmethod
    def _paper_score_map(papers: list[GraphPaper]) -> dict[str, float]:
        """Map paper_id AND title -> Cypher score so graph chunks can inherit it."""
        m: dict[str, float] = {}
        for p in papers:
            if p.paper_id:
                m[p.paper_id] = float(p.score)
            if p.title:
                m[p.title] = float(p.score)
        return m

    def answer(
        self,
        query: str,
        top_k: int = 5,
        top_papers: int = 5,
        chunks_per_paper: int = 3,
        alpha: float | None = None,
        mode: str = "hybrid",
        rerank: bool | None = None,
    ) -> GraphRAGAnswer:
        """Answer a query.

        mode:
            "hybrid"  -> graph chunks blended with D2 hybrid top-k  (default)
            "vector"  -> D2 hybrid retriever only  (ablation baseline)
            "graph"   -> graph-selected chunks only (ablation)
        """
        if is_risky_query(query):
            return GraphRAGAnswer(
                query=query,
                mode="blocked_by_safety",
                answer=(
                    "I can't answer this because the query asks the system to ignore "
                    "its sources or bypass safety rules."
                ),
                citations=[],
                warning="Risky prompt pattern detected.",
            )

        do_rerank = self.rerank if rerank is None else rerank
        pool = self.rerank_pool or max(top_k * 4, top_k)

        graph_chunks: list[SupportingChunk] = []
        paper_scores: dict[str, float] = {}
        selected_papers: list[str] = []
        selected_topics: list[str] = []
        selected_authors: list[str] = []
        warning: str | None = None

        if mode in ("hybrid", "graph"):
            selection = self.selector.select_with_chunks(
                query=query,
                top_papers=top_papers,
                chunks_per_paper=chunks_per_paper,
            )
            graph_chunks = filter_safe_chunks(selection.chunks)   # source pinning
            paper_scores = self._paper_score_map(selection.papers)
            selected_papers = selection.paper_titles
            selected_topics = [t.name for t in selection.topics]
            selected_authors = [a.name for a in selection.authors]
            warning = selection.warning

        vector_results: list[dict[str, Any]] = []
        if mode in ("hybrid", "vector"):
            vector_results = self._vector_search(
                query, top_k=max(top_k * 3, top_k), alpha=alpha
            )

        if mode == "vector":
            blended = blend_results(
                vector_results, [], {},
                vector_weight=1.0, graph_weight=0.0,
                presence_bonus=0.0, top_k=pool,
            )
            used_mode = "vector_only"
        elif mode == "graph":
            blended = blend_results(
                [], graph_chunks, paper_scores,
                vector_weight=0.0, graph_weight=1.0,
                presence_bonus=0.0, top_k=pool,
            )
            used_mode = "graph_only"
        else:
            blended = blend_results(
                vector_results, graph_chunks, paper_scores,
                vector_weight=self.vector_weight,
                graph_weight=self.graph_weight,
                presence_bonus=self.presence_bonus,
                top_k=pool,
            )
            used_mode = "graph_guided_hybrid_blend"

        # ---- rerank the blended candidate pool, then keep top_k -----------
        if do_rerank and blended:
            scorer = self._get_rerank_scorer()
            if scorer is not None:
                blended = rerank_results(query, blended, score_fn=scorer, top_k=top_k)
                used_mode = used_mode + "+rerank"
            else:
                blended = blended[:top_k]
        else:
            blended = blended[:top_k]

        if not blended:
            return GraphRAGAnswer(
                query=query,
                mode=used_mode,
                answer=(
                    "No supporting evidence was found from the graph subgraph or the "
                    "hybrid retriever, so no grounded answer can be produced."
                ),
                citations=[],
                blended=[],
                selected_papers=selected_papers,
                selected_topics=selected_topics,
                selected_authors=selected_authors,
                warning=warning or "No evidence after blending.",
            )

        # Build the cited answer string (title + page range per source).
        answer_text, _refs = build_answer_with_citations(query, blended)
        citations = self._make_citations(blended)

        return GraphRAGAnswer(
            query=query,
            mode=used_mode,
            answer=answer_text,
            citations=citations,
            blended=blended,
            selected_papers=selected_papers,
            selected_topics=selected_topics,
            selected_authors=selected_authors,
            warning=warning,
        )

    @staticmethod
    def _make_grounded_answer(query: str, blended: list[dict[str, Any]]) -> str:
        """Deprecated: kept for backward compatibility.

        Answer rendering now lives in the module-level
        ``build_answer_with_citations`` (title + page range per source).
        """
        answer_text, _ = build_answer_with_citations(query, blended)
        return answer_text

    @staticmethod
    def _make_citations(blended: list[dict[str, Any]]) -> list[dict[str, Any]]:
        citations: list[dict[str, Any]] = []
        for d in blended:
            prov = d.get("provenance") or {}
            citations.append(
                {
                    "chunk_id": d.get("chunk_id"),
                    "doc_id": d.get("doc_id"),
                    "title": d.get("title"),
                    "page_start": d.get("page_start"),
                    "page_end": d.get("page_end"),
                    "page_range": d.get("page_range") or prov.get("page_range"),
                    "source_pdf": prov.get("source_pdf"),
                    "blend_score": round(float(d.get("blend_score", 0.0)), 4),
                    "in_graph": d.get("in_graph", False),
                    "in_vector": d.get("in_vector", False),
                }
            )
        return citations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the D3 GraphRAG executor (graph subgraph blended with D2 hybrid search)."
    )
    parser.add_argument("query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of blended results to return.")
    parser.add_argument("--top-papers", type=int, default=5, help="Papers to pull from the graph.")
    parser.add_argument("--chunks-per-paper", type=int, default=3)
    parser.add_argument(
        "--alpha", type=float, default=None,
        help="BM25 weight for the D2 hybrid retriever (D1 convention; None = retriever default).",
    )
    parser.add_argument(
        "--mode", choices=["hybrid", "vector", "graph"], default="hybrid",
        help="hybrid = blend (default); vector / graph = ablation baselines.",
    )
    parser.add_argument(
        "--vector-weight", type=float, default=0.6, help="Blend weight for hybrid retriever signal.",
    )
    parser.add_argument(
        "--graph-weight", type=float, default=0.4, help="Blend weight for graph signal.",
    )
    parser.add_argument(
        "--no-rerank", action="store_true",
        help="Disable the cross-encoder rerank step (keep blend ordering).",
    )
    parser.add_argument(
        "--rerank-model", default=DEFAULT_RERANK_MODEL, help="Cross-encoder model name.",
    )
    parser.add_argument(
        "--rerank-pool", type=int, default=None,
        help="Candidate pool size before rerank (default: top_k * 4).",
    )
    args = parser.parse_args()

    with GraphRAGExecutor(
        verbose=True,
        vector_weight=args.vector_weight,
        graph_weight=args.graph_weight,
        rerank=not args.no_rerank,
        rerank_model=args.rerank_model,
        rerank_pool=args.rerank_pool,
    ) as executor:
        result = executor.answer(
            args.query,
            top_k=args.top_k,
            top_papers=args.top_papers,
            chunks_per_paper=args.chunks_per_paper,
            alpha=args.alpha,
            mode=args.mode,
        )

    print("\n========== GRAPHRAG ANSWER ==========")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
