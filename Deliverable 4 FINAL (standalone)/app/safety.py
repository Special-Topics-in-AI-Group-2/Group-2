"""safety.py — simple D3 safety mitigations.

Mitigations included:
1. Deny risky prompt-injection style queries.
2. Source pinning: only allow chunks that include text and source provenance.
3. No unsupported answer: GraphRAG executor returns no answer if no supporting chunks exist.
"""

from __future__ import annotations

from typing import Iterable


RISKY_PATTERNS = [
    "ignore previous instructions",
    "ignore the documents",
    "ignore sources",
    "do not cite",
    "without citations",
    "bypass",
    "jailbreak",
    "developer message",
    "system prompt",
]


def is_risky_query(query: str) -> bool:
    q = query.lower()
    return any(pattern in q for pattern in RISKY_PATTERNS)


def filter_safe_chunks(chunks: Iterable) -> list:
    """Keep only chunks that have actual text and provenance/page information."""
    safe = []
    for chunk in chunks:
        text = getattr(chunk, "text", "")
        provenance = getattr(chunk, "provenance", {}) or {}
        has_page = bool(
            provenance.get("page_range")
            or getattr(chunk, "page_start", None) is not None
            or getattr(chunk, "page_end", None) is not None
        )
        if text and text.strip() and has_page:
            safe.append(chunk)
    return safe
