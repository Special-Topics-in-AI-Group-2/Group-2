"""Smoke tests for the D3 GraphRAG blend/rerank pure functions.

graphrag_executor imports graph_selector, which hard-requires neo4j + pymongo,
so we skip cleanly when those service libraries are not installed.
"""

import pytest

pytest.importorskip("neo4j")
pytest.importorskip("pymongo")

import graphrag_executor as ge
from graph_selector import SupportingChunk


def _supporting(chunk_id, title, text="some evidence text", page=3):
    return SupportingChunk(
        chunk_id=chunk_id, doc_id=title, title=title, authors="", year=2020,
        venue="", page_start=page, page_end=page, chunk_index=0, text=text,
        provenance={"page_range": f"p. {page}"},
    )


def test_min_max_norm_basic():
    assert ge._min_max_norm([]) == []
    assert ge._min_max_norm([5.0]) == [1.0]
    assert ge._min_max_norm([0.0, 5.0, 10.0]) == [0.0, 0.5, 1.0]


def test_blend_presence_bonus_rewards_agreement():
    vec = [{"chunk_id": "c1", "title": "Paper A", "text": "t", "hybrid_score": 1.0,
            "page_start": 1, "page_end": 1}]
    graph = [_supporting("c1", "Paper A")]   # same chunk in both signals
    out = ge.blend_results(vec, graph, {"Paper A": 1.0},
                           vector_weight=0.6, graph_weight=0.4, presence_bonus=0.1, top_k=5)
    top = out[0]
    assert top["chunk_id"] == "c1"
    assert top["in_graph"] and top["in_vector"]
    # vector_norm(1.0) + graph_norm(1.0) + bonus = 0.6 + 0.4 + 0.1
    assert top["blend_score"] == pytest.approx(1.1, abs=1e-6)


def test_blend_vector_only_mode():
    vec = [
        {"chunk_id": "c1", "title": "A", "text": "t", "hybrid_score": 1.0, "page_start": 1, "page_end": 1},
        {"chunk_id": "c2", "title": "B", "text": "t", "hybrid_score": 0.0, "page_start": 1, "page_end": 1},
    ]
    out = ge.blend_results(vec, [], {}, vector_weight=1.0, graph_weight=0.0, presence_bonus=0.0, top_k=5)
    assert out[0]["chunk_id"] == "c1"
    assert all(not d["in_graph"] for d in out)


def test_rerank_without_scorer_preserves_order_and_truncates():
    results = [{"text": f"t{i}", "chunk_id": str(i)} for i in range(5)]
    out = ge.rerank_results("q", results, score_fn=None, top_k=3)
    assert [r["chunk_id"] for r in out] == ["0", "1", "2"]


def test_rerank_with_scorer_reorders():
    results = [{"text": "a", "chunk_id": "a"}, {"text": "b", "chunk_id": "b"}]
    out = ge.rerank_results("q", results, score_fn=lambda q, texts: [0.1, 0.9], top_k=2)
    assert out[0]["chunk_id"] == "b"


def test_build_answer_with_citations_has_sources_block():
    chunks = [{"chunk_id": "c1", "title": "Paper A", "text": "evidence sentence.",
               "page_start": 2, "page_end": 3, "provenance": {"page_range": "pp. 2-3"}}]
    answer, refs = ge.build_answer_with_citations("q", chunks)
    assert "Sources:" in answer
    assert refs[0]["title"] == "Paper A"
    assert refs[0]["page_range"] == "pp. 2-3"
