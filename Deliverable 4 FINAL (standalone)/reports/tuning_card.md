# D4 Tuning Card — PEFT/QLoRA Answer SLM

This card documents the small-language-model fine-tune that powers the `tuned`
answer backend (`app/slm.py`), produced by `app/train_slm.py`. It follows the
brief's required fields: dataset size, epochs, learning rate, LoRA ranks,
hardware/time, and license.

---

## 1. Objective

Teach a small (1–3 B) instruction model to write **faithful, `[n]`-cited**
answers in the exact GraphRAG prompt format, using *only* the retrieved sources.
The adapter changes the *prose*; the citation list and page ranges always come
from the retrieved chunks, so provenance can never be hallucinated.

## 2. Recommended configuration (production)

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen2.5-1.5B-Instruct` (1.5 B, Apache-2.0) |
| Method | **QLoRA** — 4-bit NF4 base + LoRA adapters, double quantization |
| Compute dtype | bfloat16 |
| LoRA rank `r` / `alpha` / dropout | 16 / 32 / 0.05 |
| LoRA target modules | `q_proj, k_proj, v_proj, o_proj` |
| Trainable params | ≈ 0.4–1 % of total (adapter only; base frozen) |
| Dataset | 321 curated, corpus-grounded Q/A (273 train / 48 val) from `build_qa_dataset.py` |
| Epochs / LR / batch / grad-accum | 3 / 2e-4 / 1 / 8 (effective batch 8) |
| Max sequence length | 512 |
| Optimizer | AdamW (paged AdamW 8-bit on GPU) |
| Hardware (target) | 1× consumer GPU (≥ 8 GB, e.g. RTX 3060/T4); ~2–5 min |
| Quantization at inference | 4-bit on GPU; fp32 on CPU; answer cache for latency |

Reproduce:

```bash
pip install -r requirements-slm.txt           # + bitsandbytes on a CUDA box
cd app && python train_slm.py --base-model Qwen/Qwen2.5-1.5B-Instruct --epochs 3
```

`train_slm.py` auto-detects the GPU + bitsandbytes and switches on 4-bit QLoRA
(`BitsAndBytesConfig(load_in_4bit, nf4, double_quant)` +
`prepare_model_for_kbit_training`). On CPU it transparently drops to plain LoRA.

## 3. CPU smoke validation (this repo, reproduced)

To prove the pipeline runs end-to-end without a GPU, we trained a tiny
stand-in model. These are **real** numbers from `artifacts/slm_tuning_card.json`:

| Field | Value |
|---|---|
| Base model | `sshleifer/tiny-gpt2` (pipeline stand-in) |
| Method | LoRA (no quantization — CPU) |
| LoRA `r`/`alpha`/dropout | 16 / 32 / 0.05, target `c_attn` |
| Dataset | 273 train / 48 val |
| Epochs / LR | 3 / 2e-4 |
| Trainable params | 256 (0.249 % of 102,970) |
| Hardware | CPU, Windows 11 |
| Wall time | ≈ 45 s |

The stand-in has random-quality weights, so its generated text is not
meaningful — its only purpose is to validate data flow, LoRA wiring, adapter
save/load, and the eval harness. The quality comparison below makes this
explicit.

## 4. Quality / latency comparison (final eval)

From `python app/eval_slm.py --backends extractive base tuned` (offline abstract
retriever; deterministic lexical metrics). CPU-smoke run with `tiny-gpt2`:

| Backend | Faithfulness | Answer relevance | Mean ms | p95 ms | Cache |
|---|---|---|---|---|---|
| extractive | **0.320** | **0.188** | 0.1 | 0.1 | — |
| base (zero-shot) | 0.000 | 0.000 | 601* | 735* | 1 hit |
| tuned (LoRA) | 0.000 | 0.000 | 636* | 778* | 1 hit |

\* uncached CPU generation (~0.6 s/query for the tiny stand-in over the 102-paper
corpus). The disk answer-cache brings repeated queries to ~1–2 ms (verified by
`tests/test_slm.py` cache round-trip) — the latency win the brief asks us to
demonstrate.

**Reading the table.** With a degenerate stand-in model the grounded
`extractive` backend dominates on faithfulness — which is exactly why it is the
shipped default and the faithfulness floor. With the recommended
`Qwen2.5-1.5B-Instruct` base, we expect `tuned` to (a) match `extractive`
faithfulness because the prompt pins the answer to sources, while (b) improving
fluency/answer-relevance over `base`, at higher latency that the cache absorbs
on repeats. The harness is model-agnostic: swap `--base-model` and rerun to
populate the production row.

## 5. License & ethics

* **Adapter only.** We train and ship a LoRA adapter; base-model weights remain
  under their own license (Qwen 2.5 = Apache-2.0; Llama variants = Meta
  community license). The adapter is a derivative governed by the base license.
* **Training data.** Derived from open-access arXiv papers (see
  `data/corpus_metadata.json` for per-paper DOI + license). No private or
  copyrighted full texts are redistributed in the repo — PDFs are fetched at
  run time.
* **Grounding.** Citations and page ranges are always taken from retrieved
  chunks, never generated, bounding hallucination risk.
