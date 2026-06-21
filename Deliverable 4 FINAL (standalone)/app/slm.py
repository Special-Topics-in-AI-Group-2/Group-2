"""slm.py — Deliverable 4 small-language-model answer generator.

This is the SLM that the brief asks D4 to fine-tune (PEFT/QLoRA) and integrate
into the GraphRAG pipeline.  It exposes one clean interface — ``AnswerGenerator``
— with three interchangeable backends so the *same* GraphRAG executor can run an
A/B comparison without any code change:

    backend="extractive"  no model at all.  Stitches a grounded, cited answer
                          straight from the retrieved chunks.  Always available,
                          0-dependency, the safe CPU/offline default.  This is
                          also the D3 baseline answer renderer.

    backend="base"        zero-shot generation with the *untuned* base model
                          (e.g. Qwen2.5-1.5B-Instruct).  Used as the "before"
                          column of the D4 quality/latency table.

    backend="tuned"       base model + the LoRA/QLoRA adapter trained by
                          train_slm.py.  The "after" column.

Faithfulness by construction
----------------------------
For the model backends the prompt *pins* the answer to the retrieved context and
forbids outside knowledge, and the citation list returned to the caller always
comes from the retrieved chunks — never from the model — so page-ranged
provenance stays correct even if the prose is generated.

Performance: quantize & cache
-----------------------------
* Quantization — when a CUDA GPU + bitsandbytes are present the model loads in
  4-bit (QLoRA-style) via ``BitsAndBytesConfig``; otherwise it loads in the best
  available dtype on CPU.  Either way the public API is identical.
* Caching — every (backend, model, query, contexts) tuple is hashed and the
  generated answer is memoised to ``artifacts/slm_cache/``.  Repeated demo
  queries return in ~0 ms, which is exactly the latency win the brief asks us to
  show.

Everything degrades gracefully: if torch/transformers/peft are missing or a
model fails to load, the generator logs a warning and falls back to
``extractive`` so the whole agent still runs.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # config is optional so slm.py stays importable in isolation
    from config import ARTIFACTS_DIR, SLM_ADAPTER_DIR, SLM_BASE_MODEL, SLM_BACKEND
except Exception:  # noqa: BLE001
    ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
    SLM_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
    SLM_ADAPTER_DIR = str(ARTIFACTS_DIR / "slm_lora")
    SLM_BACKEND = "extractive"


# ---------------------------------------------------------------------------
# Prompt construction (shared by base + tuned backends)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a careful scientific question-answering assistant. "
    "Answer ONLY using the numbered sources provided. "
    "Do not use outside knowledge. If the sources do not contain the answer, "
    "say so. Keep the answer to 2-4 sentences and cite supporting sources "
    "inline using their [n] markers."
)


def format_contexts(contexts: list[dict[str, Any]], max_chars: int = 500) -> str:
    """Render retrieved chunks as a numbered, citeable source block."""
    lines: list[str] = []
    for i, ch in enumerate(contexts, start=1):
        title = ch.get("title") or "Unknown Paper"
        page = ch.get("page_range") or (ch.get("provenance") or {}).get("page_range") or ""
        text = " ".join((ch.get("text") or "").split())[:max_chars]
        head = f"[{i}] {title}" + (f" ({page})" if page else "")
        lines.append(f"{head}\n{text}")
    return "\n\n".join(lines)


def build_prompt(query: str, contexts: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Build a chat-format prompt pinned to the retrieved sources."""
    user = (
        f"Sources:\n{format_contexts(contexts)}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the sources above and cite them with [n] markers."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Extractive backend (no model) — also the grounded fallback
# ---------------------------------------------------------------------------

def extractive_answer(query: str, contexts: list[dict[str, Any]], max_snippets: int = 4) -> str:
    """Deterministic grounded answer built straight from retrieved chunks.

    Mirrors D3's ``build_answer_with_citations`` body so the extractive backend
    and the legacy renderer agree.  Every sentence is traceable to a (title,
    page-range) pair, so faithfulness is 1.0 by construction.
    """
    if not contexts:
        return ("No supporting evidence with citable page ranges was found, so no "
                "grounded answer can be produced.")

    used = contexts[:max_snippets]
    lines = [f"Answer to: {query}", "", "Based on the retrieved sources:", ""]
    for i, ch in enumerate(used, start=1):
        title = ch.get("title") or "Unknown Paper"
        page = ch.get("page_range") or (ch.get("provenance") or {}).get("page_range") or "page unknown"
        snippet = " ".join((ch.get("text") or "").split())[:300].rstrip()
        if not snippet:
            continue
        if not snippet.endswith((".", "!", "?")):
            snippet += "..."
        lines.append(f"- {snippet} [{i}] ({title}, {page})")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config + generator
# ---------------------------------------------------------------------------

@dataclass
class SLMConfig:
    backend: str = SLM_BACKEND                 # extractive | base | tuned
    base_model: str = SLM_BASE_MODEL
    adapter_dir: str = SLM_ADAPTER_DIR
    # Fewer new tokens = faster CPU generation. Override via SLM_MAX_NEW_TOKENS.
    max_new_tokens: int = int(os.getenv("SLM_MAX_NEW_TOKENS", "200"))
    temperature: float = 0.0                   # 0.0 -> greedy / deterministic
    quantize: str = "auto"                     # auto | 4bit | 8bit | none
    cache_dir: Path = field(default_factory=lambda: Path(ARTIFACTS_DIR) / "slm_cache")
    device: str = "auto"


class AnswerGenerator:
    """Pluggable SLM answer generator with disk caching and graceful fallback."""

    def __init__(self, config: SLMConfig | None = None, verbose: bool = False) -> None:
        self.config = config or SLMConfig()
        self.verbose = verbose
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._effective_backend = self.config.backend
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- logging -----------------------------------------------------------
    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[slm] {msg}")

    # -- caching -----------------------------------------------------------
    def _cache_key(self, query: str, contexts: list[dict[str, Any]]) -> str:
        ctx_sig = "||".join(
            f"{c.get('chunk_id','')}:{(c.get('text') or '')[:200]}" for c in contexts
        )
        raw = f"{self._effective_backend}|{self.config.base_model}|{query}|{ctx_sig}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]

    def _cache_read(self, key: str) -> str | None:
        path = self.config.cache_dir / f"{key}.json"
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))["answer"]
            except Exception:  # noqa: BLE001
                return None
        return None

    def _cache_write(self, key: str, answer: str) -> None:
        path = self.config.cache_dir / f"{key}.json"
        try:
            path.write_text(json.dumps({"answer": answer}), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    # -- model loading -----------------------------------------------------
    def _ensure_model(self) -> bool:
        """Lazily load base (+ adapter) model.  Returns True if usable."""
        if self._loaded:
            return self._model is not None
        self._loaded = True

        if self.config.backend == "extractive":
            return False

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except Exception as exc:  # noqa: BLE001
            self._log(f"transformers/torch unavailable ({exc}); using extractive fallback.")
            self._effective_backend = "extractive"
            return False

        # ---- quantization config (QLoRA-style 4-bit when a GPU is available) ----
        quant_cfg = None
        want = self.config.quantize
        has_cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        if want in ("auto", "4bit", "8bit") and has_cuda:
            try:
                from transformers import BitsAndBytesConfig
                if want in ("auto", "4bit"):
                    quant_cfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                else:
                    quant_cfg = BitsAndBytesConfig(load_in_8bit=True)
                self._log(f"loading in {('4bit' if want in ('auto','4bit') else '8bit')} (bitsandbytes).")
            except Exception as exc:  # noqa: BLE001
                self._log(f"bitsandbytes unavailable ({exc}); loading unquantized.")
                quant_cfg = None

        dtype = torch.bfloat16 if has_cuda else torch.float32
        device_map = "auto" if has_cuda else None

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.config.base_model)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                quantization_config=quant_cfg,
                torch_dtype=dtype,
                device_map=device_map,
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"could not load base model '{self.config.base_model}' ({exc}); extractive fallback.")
            self._effective_backend = "extractive"
            self._model = None
            return False

        # ---- attach LoRA adapter for the tuned backend ----
        if self.config.backend == "tuned":
            adapter = Path(self.config.adapter_dir)
            if adapter.exists():
                try:
                    from peft import PeftModel
                    self._model = PeftModel.from_pretrained(self._model, str(adapter))
                    self._log(f"loaded LoRA adapter from {adapter}.")
                except Exception as exc:  # noqa: BLE001
                    self._log(f"adapter load failed ({exc}); using base weights only.")
                    self._effective_backend = "base"
            else:
                self._log(f"adapter dir {adapter} not found; using base weights (run train_slm.py first).")
                self._effective_backend = "base"

        if device_map is None and self._model is not None:
            self._model = self._model.to("cpu")
        if self._model is not None:
            self._model.eval()
        return self._model is not None

    # -- generation --------------------------------------------------------
    def _generate_with_model(self, query: str, contexts: list[dict[str, Any]]) -> str:
        import torch

        messages = build_prompt(query, contexts)
        tok = self._tokenizer
        # Render the prompt to text first (works for models with or without a
        # chat template), then tokenize normally.  This always yields a
        # BatchEncoding with both input_ids and a matching attention_mask —
        # avoiding the apply_chat_template return-type ambiguity (tensor vs
        # BatchEncoding) across transformers versions.
        try:
            prompt_text = tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
        except Exception:  # noqa: BLE001 — model without a chat template
            prompt_text = "\n".join(m["content"] for m in messages) + "\nAnswer:"
        enc = tok(prompt_text, return_tensors="pt")
        enc = {k: v.to(self._model.device) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[-1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.config.max_new_tokens,
            "pad_token_id": tok.pad_token_id or tok.eos_token_id,
        }
        if self.config.temperature and self.config.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.config.temperature)
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            out = self._model.generate(**enc, **gen_kwargs)
        text = tok.decode(out[0][input_len:], skip_special_tokens=True)
        return text.strip()

    def generate(self, query: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        """Generate an answer.  Returns {answer, backend, latency_ms, cached}."""
        t0 = time.perf_counter()

        # extractive needs no cache (it's already ~0 ms and deterministic)
        if self.config.backend == "extractive":
            ans = extractive_answer(query, contexts)
            return {"answer": ans, "backend": "extractive",
                    "latency_ms": (time.perf_counter() - t0) * 1000, "cached": False}

        usable = self._ensure_model()
        if not usable:
            ans = extractive_answer(query, contexts)
            return {"answer": ans, "backend": self._effective_backend,
                    "latency_ms": (time.perf_counter() - t0) * 1000, "cached": False}

        key = self._cache_key(query, contexts)
        cached = self._cache_read(key)
        if cached is not None:
            return {"answer": cached, "backend": self._effective_backend,
                    "latency_ms": (time.perf_counter() - t0) * 1000, "cached": True}

        ans = self._generate_with_model(query, contexts)
        if not ans:
            ans = extractive_answer(query, contexts)
        self._cache_write(key, ans)
        return {"answer": ans, "backend": self._effective_backend,
                "latency_ms": (time.perf_counter() - t0) * 1000, "cached": False}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_generator(backend: str | None = None, verbose: bool = False, **kw: Any) -> AnswerGenerator:
    """Build an AnswerGenerator from config, overriding the backend if given."""
    cfg = SLMConfig(backend=backend or SLM_BACKEND, **kw)
    return AnswerGenerator(cfg, verbose=verbose)


if __name__ == "__main__":
    # Tiny smoke test that needs no model and no services.
    demo_ctx = [
        {"chunk_id": "p001_c0", "title": "Attention Is All You Need", "page_range": "pp. 1-2",
         "text": "We propose the Transformer, based solely on attention mechanisms, "
                 "dispensing with recurrence and convolutions entirely."},
    ]
    gen = get_generator(backend="extractive", verbose=True)
    out = gen.generate("What is the Transformer based on?", demo_ctx)
    print(json.dumps(out, indent=2))
