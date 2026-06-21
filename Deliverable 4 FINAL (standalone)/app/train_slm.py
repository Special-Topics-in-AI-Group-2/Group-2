"""train_slm.py — Deliverable 4 PEFT/QLoRA fine-tuning of the answer SLM.

Fine-tunes a small (1-3B) instruction model on the curated, corpus-grounded Q/A
set (build_qa_dataset.py) so it produces faithful, [n]-cited answers in our exact
GraphRAG prompt format.

Design choices
--------------
* PEFT/LoRA via the ``peft`` library — only low-rank adapters are trained, so the
  footprint is a few MB and the base weights stay frozen.
* QLoRA: when a CUDA GPU + bitsandbytes are available the base model is loaded in
  4-bit (nf4, double-quant) and ``prepare_model_for_kbit_training`` is applied —
  this is the configuration documented in the tuning card.  On a CPU-only machine
  the script transparently drops to plain LoRA on a small model so it still runs
  end-to-end (used for the smoke test in CI).
* A self-contained torch training loop (AdamW) keeps the dependency surface to
  torch + transformers + peft.  TRL's ``SFTTrainer`` is the production
  alternative and is noted in the tuning card.

Outputs
-------
  artifacts/slm_lora/              the trained LoRA adapter (load via slm.py tuned backend)
  artifacts/slm_tuning_card.json   dataset size, epochs, lr, LoRA ranks, hardware, time, license

Usage
-----
  # Real run (GPU recommended):
  python train_slm.py --base-model Qwen/Qwen2.5-1.5B-Instruct --epochs 3

  # CPU smoke test (tiny model, proves the pipeline end-to-end):
  python train_slm.py --base-model sshleifer/tiny-gpt2 --epochs 2 --max-len 256
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:  # noqa: BLE001
    pass

from config import ARTIFACTS_DIR, QA_DIR, SLM_ADAPTER_DIR, SLM_BASE_MODEL


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_training_text(ex: dict, tokenizer) -> str:
    """Render one example to a single training string.

    Prefer the model's chat template (so training matches inference exactly);
    fall back to the pre-rendered flat ``text`` field for models without one
    (e.g. tiny-gpt2 used in the CPU smoke test).
    """
    messages = ex.get("messages")
    if messages and getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False)
        except Exception:  # noqa: BLE001
            pass
    return ex["text"]


def main() -> int:
    ap = argparse.ArgumentParser(description="PEFT/QLoRA fine-tune the answer SLM.")
    ap.add_argument("--base-model", default=SLM_BASE_MODEL)
    ap.add_argument("--adapter-dir", default=SLM_ADAPTER_DIR)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--quantize", choices=["auto", "4bit", "none"], default="auto")
    args = ap.parse_args()

    # ---- dependencies -----------------------------------------------------
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model
    except Exception as exc:  # noqa: BLE001
        print(f"[train] missing deps ({exc}).\n"
              "        Install the SLM extras:  pip install -r requirements-slm.txt")
        return 2

    train = load_jsonl(QA_DIR / "qa_train.jsonl")
    val = load_jsonl(QA_DIR / "qa_val.jsonl")
    if not train:
        print(f"[train] no training data in {QA_DIR} — run build_qa_dataset.py first.")
        return 2
    print(f"[train] base model : {args.base_model}")
    print(f"[train] train / val: {len(train)} / {len(val)}")

    has_cuda = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    device = "cuda" if has_cuda else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- base model + optional 4-bit QLoRA --------------------------------
    quant_cfg = None
    used_quant = "none"
    if args.quantize in ("auto", "4bit") and has_cuda:
        try:
            from transformers import BitsAndBytesConfig
            quant_cfg = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            used_quant = "4bit-nf4"
        except Exception as exc:  # noqa: BLE001
            print(f"[train] bitsandbytes unavailable ({exc}); training without quantization.")

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=quant_cfg,
        torch_dtype=torch.bfloat16 if has_cuda else torch.float32,
        device_map="auto" if has_cuda else None,
    )

    if quant_cfg is not None:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)

    # ---- LoRA adapter -----------------------------------------------------
    # Resolve target modules robustly across architectures (gpt2 vs llama/qwen).
    name = args.base_model.lower()
    if "gpt2" in name:
        target_modules = ["c_attn"]
    else:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"[train] trainable params: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")
    if device == "cpu":
        model = model.to("cpu")
    model.train()

    # ---- tokenize ---------------------------------------------------------
    def encode(rows: list[dict]):
        batch = []
        for ex in rows:
            text = render_training_text(ex, tokenizer)
            ids = tokenizer(text, truncation=True, max_length=args.max_len,
                            return_tensors="pt").input_ids[0]
            batch.append(ids)
        return batch

    train_ids = encode(train)

    # ---- manual training loop (AdamW + grad accumulation) -----------------
    optim = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr)
    t0 = time.perf_counter()
    step = 0
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        optim.zero_grad()
        for i, ids in enumerate(train_ids):
            ids = ids.unsqueeze(0).to(device)
            out = model(input_ids=ids, labels=ids)
            loss = out.loss / args.grad_accum
            loss.backward()
            epoch_loss += out.loss.item()
            if (i + 1) % args.grad_accum == 0:
                optim.step()
                optim.zero_grad()
                step += 1
        optim.step()
        optim.zero_grad()
        print(f"[train] epoch {epoch+1}/{args.epochs}  mean_loss={epoch_loss/len(train_ids):.4f}")
    elapsed = time.perf_counter() - t0

    # ---- save adapter -----------------------------------------------------
    adapter_dir = Path(args.adapter_dir)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"[train] saved LoRA adapter -> {adapter_dir}")

    # ---- tuning card ------------------------------------------------------
    card = {
        "base_model": args.base_model,
        "method": "QLoRA (4-bit nf4)" if used_quant.startswith("4bit") else "LoRA",
        "quantization": used_quant,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout,
                 "target_modules": target_modules},
        "dataset": {"train_examples": len(train), "val_examples": len(val),
                    "source": "curated corpus Q/A (build_qa_dataset.py)"},
        "hyperparameters": {"epochs": args.epochs, "lr": args.lr,
                            "batch_size": args.batch_size, "grad_accum": args.grad_accum,
                            "max_seq_len": args.max_len},
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 4),
        "hardware": {"device": device,
                     "gpu": torch.cuda.get_device_name(0) if has_cuda else None,
                     "platform": platform.platform()},
        "wall_time_seconds": round(elapsed, 2),
        "adapter_dir": str(adapter_dir),
        "license_note": (
            "Base model weights are subject to their own model license (e.g. Qwen / "
            "Llama community licenses). Only the LoRA adapter is produced here. "
            "Training data derived from open-access arXiv papers (see corpus_metadata.json)."
        ),
    }
    Path(ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    card_path = Path(ARTIFACTS_DIR) / "slm_tuning_card.json"
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    print(f"[train] wrote tuning card -> {card_path}")
    print(f"[train] done in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
