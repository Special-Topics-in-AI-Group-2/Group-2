"""
safety_filters.py — provenance safety filter for Deliverable 3 GraphRAG.

Purpose
-------
This file checks that every evidence chunk used in a GraphRAG answer has valid
provenance metadata before it is allowed into the final answer.

A chunk is kept only if it has:
    1. non-empty text,
    2. a real PDF filename ending with ".pdf",
    3. a valid 1-based page number or page range.

It works with both:
    - dict chunks returned by retriever / graphrag_executor.py
    - dataclass/object chunks such as SupportingChunk from graph_selector.py

Recommended use inside graphrag_executor.py:
    from safety_filters import filter_chunks_with_valid_provenance, pin_sources

    blended, dropped = filter_chunks_with_valid_provenance(
        blended,
        return_dropped=True
    )

    blended, rejected = pin_sources(
        blended,
        approved_corpus_dir="./data/pdfs",
        return_rejected=True
    )

    if not blended:
        return GraphRAGAnswer(
            query=query,
            mode=used_mode,
            answer="No answer produced because all evidence chunks failed provenance checks.",
            citations=[],
            blended=[],
            warning=f"All chunks dropped by provenance safety filter: {len(dropped)} dropped."
        )

CLI usage:
    python safety_filters.py input_answer.json
    python safety_filters.py input_answer.json --output cleaned_answer.json
    python safety_filters.py input_answer.json --chunk-field blended
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

INVALID_FILENAME_VALUES = {
    "",
    "unknown",
    "unknown.pdf",
    "none",
    "null",
    "n/a",
    "na",
    "source unknown",
    "document unknown",
}


CHUNK_FIELDS = (
    "blended",
    "chunks",
    "evidence",
    "contexts",
    "supporting_chunks",
    "retrieved_chunks",
)


INJECTION_PATTERNS = (
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"forget\s+(all\s+)?(previous|prior|above)\s+instructions",
    r"override\s+(the\s+)?(system|developer|safety)\s+(prompt|instructions|rules)",
    r"reveal\s+(the\s+)?(system|developer)\s+(prompt|message|instructions)",
    r"you\s+are\s+now\s+(in\s+)?developer\s+mode",
    r"jailbreak",
    r"prompt\s+injection",
    r"system\s*:",
    r"developer\s*:",
    r"assistant\s*:",
    r"<\s*system\s*>",
    r"<\s*/\s*system\s*>",
    r"tool_call",
    r"function_call",
    r"execute\s+this\s+command",
    r"run\s+this\s+shell\s+command",
)

SUSPICIOUS_SOURCE_PATTERNS = (
    r"^\s*https?://",
    r"^\s*file://",
    r"^\s*data:",
    r"\.\.",
    r"[;&|`$<>]",
)


def discover_approved_pdfs(approved_corpus_dir: str | Path) -> set[str]:
    """Return approved PDF filenames found inside the official corpus folder.

    The returned set contains lowercase PDF filenames only, not full paths.
    This makes the check robust when chunks store only "paper.pdf" instead of
    the full local file path.
    """
    root = Path(approved_corpus_dir).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        return set()

    return {item.name.lower() for item in root.rglob("*.pdf") if item.is_file()}


def _safe_resolve_path(value: Any) -> Path | None:
    """Resolve a source path safely. Return None for URL-like/injected paths."""
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    for pattern in SUSPICIOUS_SOURCE_PATTERNS:
        if re.search(pattern, raw, flags=re.IGNORECASE):
            return None

    try:
        return Path(raw).expanduser().resolve()
    except Exception:
        return None


def _path_is_inside(child: Path, parent: Path) -> bool:
    """Return True when child is inside parent or equals parent."""
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _candidate_source_values(data: dict[str, Any]) -> list[str]:
    """Collect possible source path/filename values from a chunk."""
    values: list[str] = []

    keys = (
        "path",
        "source_path",
        "pdf_path",
        "file_path",
        "filename",
        "pdf_filename",
        "source_pdf",
        "source",
        "file",
        "file_name",
        "document",
        "doc_name",
    )

    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            values.append(str(value))

    provenance = data.get("provenance") or {}
    if isinstance(provenance, dict):
        for key in keys:
            value = provenance.get(key)
            if value not in (None, ""):
                values.append(str(value))

        metadata = provenance.get("metadata") or {}
        if isinstance(metadata, dict):
            for key in keys:
                value = metadata.get(key)
                if value not in (None, ""):
                    values.append(str(value))

    metadata = data.get("metadata") or {}
    if isinstance(metadata, dict):
        for key in keys:
            value = metadata.get(key)
            if value not in (None, ""):
                values.append(str(value))

    # Preserve order while removing duplicates.
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)

    return unique_values


def _looks_injected_or_out_of_scope(data: dict[str, Any]) -> tuple[bool, str]:
    """Detect obvious prompt-injection or out-of-scope content in a chunk.

    This is intentionally conservative. It catches hostile instructions that
    should never be treated as evidence, especially if they appear inside a
    retrieved chunk or metadata field.
    """
    text_fields = [
        str(data.get("text") or ""),
        str(data.get("content") or ""),
        str(data.get("title") or ""),
        str(data.get("source") or ""),
        str(data.get("filename") or ""),
    ]

    provenance = data.get("provenance") or {}
    if isinstance(provenance, dict):
        text_fields.extend(
            str(provenance.get(key) or "")
            for key in ("filename", "source_pdf", "source", "path", "title")
        )

    combined = "\n".join(text_fields)

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, combined, flags=re.IGNORECASE):
            return True, f"prompt_injection_pattern:{pattern}"

    for source_value in _candidate_source_values(data):
        for pattern in SUSPICIOUS_SOURCE_PATTERNS:
            if re.search(pattern, source_value, flags=re.IGNORECASE):
                return True, f"suspicious_source_pattern:{pattern}"

    return False, "ok"


def pin_sources(
    chunks: Iterable[Any],
    approved_corpus_dir: str | Path,
    allowed_pdf_filenames: Iterable[str] | None = None,
    return_rejected: bool = False,
    require_file_exists: bool = True,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Allow only chunks pinned to the approved PDF corpus folder.

    A chunk is kept only when:
        1. it passes normal provenance validation,
        2. it does not look prompt-injected,
        3. its PDF filename is inside the approved corpus allow-list,
        4. and, when a full path is provided, that path resolves inside the
           approved corpus folder.

    Args:
        chunks:
            Retrieved chunks used by GraphRAG.
        approved_corpus_dir:
            Official local folder that contains the approved PDF corpus.
            Example: "data/pdfs" or "./corpus/papers".
        allowed_pdf_filenames:
            Optional explicit allow-list. If omitted, the function scans
            approved_corpus_dir recursively for *.pdf files.
        return_rejected:
            If True, return (kept_chunks, rejected_reports).
        require_file_exists:
            If True, the PDF must exist in approved_corpus_dir. Keep True for
            final evaluation/submission. Set False only for unit tests.

    Returns:
        kept_chunks
        OR
        (kept_chunks, rejected_reports)
    """
    root = Path(approved_corpus_dir).expanduser().resolve()

    if allowed_pdf_filenames is None:
        approved_names = discover_approved_pdfs(root)
    else:
        approved_names = {
            Path(str(name)).name.lower()
            for name in allowed_pdf_filenames
            if str(name).strip().lower().endswith(".pdf")
        }

    kept: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for chunk in chunks or []:
        valid, reason, normalized = validate_chunk_provenance(chunk)

        if not valid:
            rejected.append(
                {
                    "reason": reason,
                    "chunk_id": normalized.get("chunk_id"),
                    "doc_id": normalized.get("doc_id"),
                    "title": normalized.get("title"),
                    "raw_chunk": normalized,
                }
            )
            continue

        suspicious, suspicious_reason = _looks_injected_or_out_of_scope(normalized)
        if suspicious:
            rejected.append(
                {
                    "reason": suspicious_reason,
                    "chunk_id": normalized.get("chunk_id"),
                    "doc_id": normalized.get("doc_id"),
                    "title": normalized.get("title"),
                    "raw_chunk": normalized,
                }
            )
            continue

        pdf_filename = _extract_valid_pdf(normalized)
        if not pdf_filename:
            rejected.append(
                {
                    "reason": "missing_or_invalid_pdf_filename",
                    "chunk_id": normalized.get("chunk_id"),
                    "doc_id": normalized.get("doc_id"),
                    "title": normalized.get("title"),
                    "raw_chunk": normalized,
                }
            )
            continue

        pdf_name_key = Path(pdf_filename).name.lower()

        if approved_names and pdf_name_key not in approved_names:
            rejected.append(
                {
                    "reason": "pdf_not_in_approved_corpus_allowlist",
                    "pdf_filename": pdf_filename,
                    "chunk_id": normalized.get("chunk_id"),
                    "doc_id": normalized.get("doc_id"),
                    "title": normalized.get("title"),
                    "raw_chunk": normalized,
                }
            )
            continue

        if require_file_exists:
            approved_file_path = root / Path(pdf_filename).name
            if not approved_file_path.exists() and pdf_name_key not in approved_names:
                rejected.append(
                    {
                        "reason": "approved_pdf_file_not_found",
                        "pdf_filename": pdf_filename,
                        "chunk_id": normalized.get("chunk_id"),
                        "doc_id": normalized.get("doc_id"),
                        "title": normalized.get("title"),
                        "raw_chunk": normalized,
                    }
                )
                continue

        # If the chunk contains a full source path, it must not escape the corpus root.
        # A bare filename such as "paper1.pdf" is not treated as a path escape;
        # it is already controlled by the approved_names allow-list above.
        source_paths = []
        for source_value in _candidate_source_values(normalized):
            raw_source = str(source_value).strip()
            has_path_separator = "/" in raw_source or "\\" in raw_source
            is_absolute_like = Path(raw_source).is_absolute()

            if not has_path_separator and not is_absolute_like:
                continue

            resolved = _safe_resolve_path(source_value)
            if resolved and resolved.suffix.lower() == ".pdf":
                source_paths.append(resolved)

        escaped_paths = [
            str(source_path)
            for source_path in source_paths
            if source_path.is_absolute() and not _path_is_inside(source_path, root)
        ]

        if escaped_paths:
            rejected.append(
                {
                    "reason": "source_path_outside_approved_corpus_folder",
                    "pdf_filename": pdf_filename,
                    "escaped_paths": escaped_paths,
                    "chunk_id": normalized.get("chunk_id"),
                    "doc_id": normalized.get("doc_id"),
                    "title": normalized.get("title"),
                    "raw_chunk": normalized,
                }
            )
            continue

        normalized.setdefault("provenance", {})
        normalized["provenance"]["pinned_source"] = True
        normalized["provenance"]["approved_corpus_dir"] = str(root)
        kept.append(normalized)

    if return_rejected:
        return kept, rejected
    return kept



def _to_plain_dict(obj: Any) -> dict[str, Any]:
    """Convert dataclass/object/dict chunks into a normal dictionary."""
    if isinstance(obj, dict):
        return dict(obj)

    if is_dataclass(obj):
        return asdict(obj)

    data: dict[str, Any] = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        try:
            value = getattr(obj, key)
        except Exception:
            continue
        if callable(value):
            continue
        data[key] = value
    return data


def _nested_get(data: dict[str, Any], *path: str) -> Any:
    """Safely read nested dictionary values."""
    current: Any = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first_present(data: dict[str, Any], keys: Iterable[str]) -> Any:
    """Return the first non-empty value from top level or provenance."""
    provenance = data.get("provenance") or {}

    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value

    if isinstance(provenance, dict):
        for key in keys:
            value = provenance.get(key)
            if value not in (None, ""):
                return value

    return None


def _extract_pdf_filename(value: Any) -> str | None:
    """Return a clean PDF filename from a filename/path-like value."""
    if value is None:
        return None

    raw = str(value).strip()
    if not raw:
        return None

    # Accept both normal paths and URLs. Path(...).name handles normal paths;
    # regex fallback handles URL-style strings.
    candidate = Path(raw).name.strip()
    if not candidate or ".pdf" not in candidate.lower():
        match = re.search(r"([^/\\?#]+\.pdf)(?:[?#].*)?$", raw, flags=re.IGNORECASE)
        candidate = match.group(1).strip() if match else candidate

    candidate = candidate.strip()
    if candidate.lower() in INVALID_FILENAME_VALUES:
        return None

    if not candidate.lower().endswith(".pdf"):
        return None

    # Avoid accepting meaningless placeholders like ".pdf".
    stem = Path(candidate).stem.strip()
    if not stem or stem.lower() in INVALID_FILENAME_VALUES:
        return None

    return candidate


def _parse_positive_int(value: Any) -> int | None:
    """Parse a 1-based positive page number."""
    if value is None or value == "":
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, int):
        return value if value >= 1 else None

    if isinstance(value, float):
        if value.is_integer() and value >= 1:
            return int(value)
        return None

    raw = str(value).strip()
    if not raw:
        return None

    if raw.isdigit():
        number = int(raw)
        return number if number >= 1 else None

    return None


def _extract_page_from_range(value: Any) -> int | None:
    """Extract first valid page from strings/lists/dicts like 'pp. 3-4'."""
    if value is None or value == "":
        return None

    if isinstance(value, (list, tuple)) and value:
        for item in value:
            page = _parse_positive_int(item)
            if page is not None:
                return page
        return None

    if isinstance(value, dict):
        for key in ("page_start", "start", "from", "page", "page_number"):
            page = _parse_positive_int(value.get(key))
            if page is not None:
                return page
        return None

    raw = str(value).strip()
    if not raw:
        return None

    # Handles: "p. 5", "page 5", "pp. 3-4", "3-4", "pages 10 to 12"
    numbers = re.findall(r"\d+", raw)
    for token in numbers:
        page = _parse_positive_int(token)
        if page is not None:
            return page

    return None


def _extract_valid_page(data: dict[str, Any]) -> int | None:
    """Find a valid page from top-level fields or provenance fields."""
    direct_candidates = (
        "page_start",
        "page",
        "page_number",
        "page_num",
        "pageIndex",
    )

    for key in direct_candidates:
        page = _parse_positive_int(data.get(key))
        if page is not None:
            return page

    provenance = data.get("provenance") or {}
    if isinstance(provenance, dict):
        for key in direct_candidates:
            page = _parse_positive_int(provenance.get(key))
            if page is not None:
                return page

    range_candidates = (
        data.get("page_range"),
        data.get("pages"),
        data.get("page_end"),
        _nested_get(data, "provenance", "page_range"),
        _nested_get(data, "provenance", "pages"),
        _nested_get(data, "provenance", "page_end"),
    )

    for value in range_candidates:
        page = _extract_page_from_range(value)
        if page is not None:
            return page

    return None


def _extract_valid_pdf(data: dict[str, Any]) -> str | None:
    """Find a valid PDF filename from common metadata fields."""
    pdf_value = _first_present(
        data,
        (
            "filename",
            "pdf_filename",
            "source_pdf",
            "source",
            "file",
            "file_name",
            "document",
            "doc_name",
            "path",
        ),
    )

    filename = _extract_pdf_filename(pdf_value)
    if filename:
        return filename

    # Sometimes source PDF is hidden deeper inside provenance/source_metadata.
    for path in (
        ("provenance", "metadata", "filename"),
        ("provenance", "metadata", "source_pdf"),
        ("metadata", "filename"),
        ("metadata", "source_pdf"),
    ):
        filename = _extract_pdf_filename(_nested_get(data, *path))
        if filename:
            return filename

    return None


# ---------------------------------------------------------------------------
# Public validation/filter API
# ---------------------------------------------------------------------------

def validate_chunk_provenance(chunk: Any) -> tuple[bool, str, dict[str, Any]]:
    """Validate one chunk.

    Returns:
        (is_valid, reason, normalized_chunk_dict)

    reason is "ok" when valid. For invalid chunks, reason explains why it was
    dropped.
    """
    data = _to_plain_dict(chunk)

    text = str(data.get("text") or data.get("content") or "").strip()
    if not text:
        return False, "missing_text", data

    pdf_filename = _extract_valid_pdf(data)
    if not pdf_filename:
        return False, "missing_or_invalid_pdf_filename", data

    page_number = _extract_valid_page(data)
    if page_number is None:
        return False, "missing_or_invalid_page_number", data

    # Normalize provenance so later answer/citation code can use one shape.
    provenance = data.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}

    provenance.setdefault("filename", pdf_filename)
    provenance.setdefault("source_pdf", pdf_filename)

    if not provenance.get("page_start"):
        provenance["page_start"] = page_number

    if not provenance.get("page_range"):
        page_end = _parse_positive_int(data.get("page_end")) or _parse_positive_int(
            provenance.get("page_end")
        )
        if page_end and page_end != page_number:
            provenance["page_range"] = f"pp. {page_number}-{page_end}"
        else:
            provenance["page_range"] = f"p. {page_number}"

    data["provenance"] = provenance
    data.setdefault("page_start", page_number)
    data.setdefault("page_range", provenance["page_range"])

    return True, "ok", data


def has_valid_provenance(chunk: Any) -> bool:
    """Return True only when the chunk has valid PDF + page provenance."""
    valid, _, _ = validate_chunk_provenance(chunk)
    return valid


def filter_chunks_with_valid_provenance(
    chunks: Iterable[Any],
    return_dropped: bool = False,
) -> list[dict[str, Any]] | tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop chunks that do not have valid provenance metadata.

    Args:
        chunks: list/iterable of dicts, dataclasses, or simple objects.
        return_dropped: when True, also returns dropped chunk reports.

    Returns:
        kept_chunks
        OR
        (kept_chunks, dropped_reports)

    dropped_reports item shape:
        {
            "reason": "missing_or_invalid_page_number",
            "chunk_id": "...",
            "title": "...",
            "raw_chunk": {...}
        }
    """
    kept: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for chunk in chunks or []:
        valid, reason, normalized = validate_chunk_provenance(chunk)
        if valid:
            kept.append(normalized)
        else:
            dropped.append(
                {
                    "reason": reason,
                    "chunk_id": normalized.get("chunk_id"),
                    "doc_id": normalized.get("doc_id"),
                    "title": normalized.get("title"),
                    "raw_chunk": normalized,
                }
            )

    if return_dropped:
        return kept, dropped
    return kept



def pin_answer_sources(
    answer_payload: dict[str, Any],
    approved_corpus_dir: str | Path,
    chunk_field: str | None = None,
    allowed_pdf_filenames: Iterable[str] | None = None,
    require_file_exists: bool = True,
) -> dict[str, Any]:
    """Apply pin_sources() to a full answer payload.

    This filters the chunk list and adds a source_pinning_report.
    """
    payload = dict(answer_payload)

    field = chunk_field
    if field is None:
        field = next(
            (
                candidate
                for candidate in CHUNK_FIELDS
                if isinstance(payload.get(candidate), list)
            ),
            None,
        )

    if not field:
        payload["source_pinning_report"] = {
            "checked_field": None,
            "kept": 0,
            "rejected": 0,
            "message": "No chunk list field found in answer payload.",
        }
        return payload

    kept, rejected = pin_sources(
        payload.get(field, []),
        approved_corpus_dir=approved_corpus_dir,
        allowed_pdf_filenames=allowed_pdf_filenames,
        return_rejected=True,
        require_file_exists=require_file_exists,
    )

    payload[field] = kept
    payload["source_pinning_report"] = {
        "checked_field": field,
        "approved_corpus_dir": str(Path(approved_corpus_dir).expanduser().resolve()),
        "kept": len(kept),
        "rejected": len(rejected),
        "rejected_reasons": _count_reasons(rejected),
        "rejected_chunks": rejected,
    }

    kept_ids = {str(ch.get("chunk_id")) for ch in kept if ch.get("chunk_id")}
    if kept_ids and isinstance(payload.get("citations"), list):
        payload["citations"] = [
            citation
            for citation in payload["citations"]
            if str(citation.get("chunk_id")) in kept_ids
        ]

    return payload


def filter_answer_payload(
    answer_payload: dict[str, Any],
    chunk_field: str | None = None,
) -> dict[str, Any]:
    """Filter chunks inside an answer/result dictionary.

    This is useful for JSON outputs from graphrag_executor.py.

    It finds a chunk list in one of:
        blended, chunks, evidence, contexts, supporting_chunks, retrieved_chunks

    Then it replaces that list with only valid chunks and adds:
        safety_filter_report
    """
    payload = dict(answer_payload)

    field = chunk_field
    if field is None:
        field = next(
            (
                candidate
                for candidate in CHUNK_FIELDS
                if isinstance(payload.get(candidate), list)
            ),
            None,
        )

    if not field:
        payload["safety_filter_report"] = {
            "checked_field": None,
            "kept": 0,
            "dropped": 0,
            "message": "No chunk list field found in answer payload.",
        }
        return payload

    kept, dropped = filter_chunks_with_valid_provenance(
        payload.get(field, []),
        return_dropped=True,
    )

    payload[field] = kept
    payload["safety_filter_report"] = {
        "checked_field": field,
        "kept": len(kept),
        "dropped": len(dropped),
        "dropped_reasons": _count_reasons(dropped),
        "dropped_chunks": dropped,
    }

    # Keep citations aligned with surviving chunks when chunk_id is available.
    kept_ids = {str(ch.get("chunk_id")) for ch in kept if ch.get("chunk_id")}
    if kept_ids and isinstance(payload.get("citations"), list):
        payload["citations"] = [
            citation
            for citation in payload["citations"]
            if str(citation.get("chunk_id")) in kept_ids
        ]

    return payload


def _count_reasons(dropped: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in dropped:
        reason = item.get("reason", "unknown")
        counts[reason] = counts.get(reason, 0) + 1
    return counts


# Backward-friendly alias if the project wants a shorter function name.
def filter_safe_chunks(chunks: Iterable[Any]) -> list[dict[str, Any]]:
    """Alias: keep only chunks with text, real PDF filename, and page provenance."""
    return filter_chunks_with_valid_provenance(chunks)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Drop GraphRAG answer chunks that do not have valid PDF/page provenance."
    )
    parser.add_argument("input_json", help="Path to a JSON answer file or list of chunks.")
    parser.add_argument(
        "--output",
        "-o",
        help="Optional path to save the cleaned JSON. If omitted, prints to stdout.",
    )
    parser.add_argument(
        "--chunk-field",
        help="Specific answer payload field to filter, e.g. blended or chunks.",
    )
    parser.add_argument(
        "--approved-corpus-dir",
        help="Optional approved PDF corpus folder. When provided, source pinning is applied.",
    )
    parser.add_argument(
        "--no-require-file-exists",
        action="store_true",
        help="For tests only: allow approved filenames even when files are not present locally.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        kept, dropped = filter_chunks_with_valid_provenance(data, return_dropped=True)

        if args.approved_corpus_dir:
            kept, rejected = pin_sources(
                kept,
                approved_corpus_dir=args.approved_corpus_dir,
                return_rejected=True,
                require_file_exists=not args.no_require_file_exists,
            )
            source_pinning_report = {
                "checked_field": "root_list",
                "approved_corpus_dir": str(Path(args.approved_corpus_dir).expanduser().resolve()),
                "kept": len(kept),
                "rejected": len(rejected),
                "rejected_reasons": _count_reasons(rejected),
                "rejected_chunks": rejected,
            }
        else:
            source_pinning_report = None

        cleaned: Any = {
            "chunks": kept,
            "safety_filter_report": {
                "checked_field": "root_list",
                "kept": len(kept),
                "dropped": len(dropped),
                "dropped_reasons": _count_reasons(dropped),
                "dropped_chunks": dropped,
            },
        }

        if source_pinning_report:
            cleaned["source_pinning_report"] = source_pinning_report

    elif isinstance(data, dict):
        cleaned = filter_answer_payload(data, chunk_field=args.chunk_field)
        if args.approved_corpus_dir:
            cleaned = pin_answer_sources(
                cleaned,
                approved_corpus_dir=args.approved_corpus_dir,
                chunk_field=args.chunk_field,
                require_file_exists=not args.no_require_file_exists,
            )
    else:
        raise SystemExit("Input JSON must be either a list of chunks or an answer dictionary.")

    output_text = json.dumps(cleaned, indent=2, ensure_ascii=False)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(output_text + "\n", encoding="utf-8")
        print(f"Cleaned output saved to: {output_path}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()
