"""Smoke tests for the D4 SLM answer generator (slm.py).

These run without torch/transformers and without any services — they exercise
the extractive backend, prompt construction, and the disk cache.
"""

import slm


CTX = [
    {"chunk_id": "p001_c0", "title": "Attention Is All You Need", "page_range": "pp. 1-2",
     "text": "We propose the Transformer, based solely on attention mechanisms, "
             "dispensing with recurrence and convolutions entirely."},
    {"chunk_id": "p008_c0", "title": "LoRA: Low-Rank Adaptation", "page_range": "p. 3",
     "text": "LoRA freezes the pretrained weights and injects trainable low-rank "
             "matrices into each layer, drastically reducing trainable parameters."},
]


def test_format_contexts_numbers_sources():
    block = slm.format_contexts(CTX)
    assert "[1] Attention Is All You Need" in block
    assert "[2] LoRA" in block


def test_build_prompt_is_pinned_to_sources():
    messages = slm.build_prompt("What is the Transformer based on?", CTX)
    assert messages[0]["role"] == "system"
    assert "only" in messages[0]["content"].lower()
    assert "Sources:" in messages[1]["content"]
    assert "Question:" in messages[1]["content"]


def test_extractive_answer_is_grounded_and_cited():
    ans = slm.extractive_answer("What is the Transformer based on?", CTX)
    assert "[1]" in ans
    assert "Attention Is All You Need" in ans


def test_extractive_answer_handles_no_context():
    ans = slm.extractive_answer("anything", [])
    assert "no" in ans.lower()


def test_generator_extractive_backend_runs_offline(tmp_path):
    cfg = slm.SLMConfig(backend="extractive", cache_dir=tmp_path / "cache")
    gen = slm.AnswerGenerator(cfg)
    out = gen.generate("What is the Transformer based on?", CTX)
    assert out["backend"] == "extractive"
    assert out["cached"] is False
    assert "[1]" in out["answer"]
    assert out["latency_ms"] >= 0


def test_cache_key_is_deterministic_and_context_sensitive(tmp_path):
    gen = slm.AnswerGenerator(slm.SLMConfig(backend="base", cache_dir=tmp_path / "c"))
    k1 = gen._cache_key("q", CTX)
    k2 = gen._cache_key("q", CTX)
    k3 = gen._cache_key("q", CTX[:1])
    assert k1 == k2
    assert k1 != k3


def test_cache_round_trip(tmp_path):
    gen = slm.AnswerGenerator(slm.SLMConfig(backend="base", cache_dir=tmp_path / "c"))
    key = gen._cache_key("q", CTX)
    assert gen._cache_read(key) is None
    gen._cache_write(key, "cached answer")
    assert gen._cache_read(key) == "cached answer"
