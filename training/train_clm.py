import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a GPT-2 style causal LM on a text corpus.")
    p.add_argument("--model", type=str, default="distilgpt2", help="Base model name (HF hub) or local path.")
    p.add_argument("--train", type=str, required=True, help="Path to train.txt")
    p.add_argument("--valid", type=str, default=None, help="Optional path to valid.txt")
    p.add_argument("--out", type=str, required=True, help="Output directory for the fine-tuned model.")

    p.add_argument("--block-size", type=int, default=256, help="Token block size for packing.")
    p.add_argument("--epochs", type=float, default=2.0, help="Number of epochs.")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")

    p.add_argument("--batch-size", type=int, default=1, help="Per-device train batch size (CPU: keep small).")
    p.add_argument("--eval-batch-size", type=int, default=1, help="Per-device eval batch size.")
    p.add_argument("--grad-accum", type=int, default=16, help="Gradient accumulation steps.")

    p.add_argument("--seed", type=int, default=1337, help="Random seed.")
    p.add_argument("--save-steps", type=int, default=500, help="Checkpoint save interval (steps).")
    p.add_argument("--logging-steps", type=int, default=25, help="Logging interval (steps).")
    p.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio.")
    p.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay.")

    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Training device selection. 'auto' uses the default Transformers behavior.",
    )

    p.add_argument("--fp16", action="store_true", help="Enable fp16 (GPU-only).")
    p.add_argument(
        "--auto-fp16",
        action="store_true",
        help="Enable fp16 automatically when CUDA is available (otherwise keep fp16 off).",
    )
    p.add_argument("--bf16", action="store_true", help="Enable bf16 (usually GPU-only).")
    p.add_argument("--num-workers", type=int, default=0, help="Dataloader workers (Windows: 0 is safest).")
    p.add_argument("--no-eval", action="store_true", help="Disable evaluation even if --valid is provided.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Keep tokenizers quiet on Windows.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
        set_seed,
    )
    import inspect

    try:
        import torch
    except Exception as e:
        raise SystemExit(f"Failed to import torch. Did you install requirements-train.txt?\n\n{e}") from e

    cuda_available = bool(torch.cuda.is_available())
    if args.device == "cuda" and (not cuda_available):
        raise SystemExit(
            "Requested --device cuda, but torch.cuda.is_available() is False.\n"
            "Install a CUDA-enabled PyTorch build and ensure your NVIDIA driver is working.\n"
            "Tip: run `nvidia-smi` and see training/README.md section 'Enable NVIDIA GPU (CUDA) on Windows'."
        )

    # Safety: fp16 should only be used on CUDA.
    if bool(args.fp16) and (not cuda_available):
        raise SystemExit("--fp16 was set but CUDA is not available. Use --device cuda with a CUDA PyTorch build.")

    effective_fp16 = bool(args.fp16) or (bool(args.auto_fp16) and cuda_available)

    print("=== device info ===")
    print(f"torch_version: {getattr(torch, '__version__', 'unknown')}")
    print(f"cuda_runtime: {getattr(torch.version, 'cuda', None)}")
    print(f"cuda_is_available: {cuda_available}")
    if cuda_available:
        try:
            print(f"cuda_device: {torch.cuda.get_device_name(0)}")
        except Exception:
            print("cuda_device: <unknown>")
    print(f"requested_device: {args.device}")
    print(f"fp16: {effective_fp16} (requested_fp16={bool(args.fp16)}, auto_fp16={bool(args.auto_fp16)})")
    print("===============")

    set_seed(int(args.seed))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer))

    data_files = {"train": str(Path(args.train))}
    if args.valid and (not args.no_eval):
        data_files["validation"] = str(Path(args.valid))

    raw = load_dataset("text", data_files=data_files)

    def tokenize_fn(batch):
        return tokenizer(batch["text"])

    tokenized = raw.map(tokenize_fn, batched=True, remove_columns=["text"])

    block_size = int(args.block_size)
    if block_size <= 0:
        raise SystemExit("--block-size must be > 0")

    def group_texts(examples):
        # Concatenate and split into fixed blocks.
        concatenated = {k: sum(examples[k], []) for k in examples.keys()}
        total_len = len(concatenated["input_ids"])
        if total_len < block_size:
            return {k: [] for k in concatenated.keys()}
        total_len = (total_len // block_size) * block_size
        result = {
            k: [t[i : i + block_size] for i in range(0, total_len, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    lm_ds = tokenized.map(group_texts, batched=True)

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Transformers' TrainingArguments has changed device flags across versions.
    # We support both styles to keep this script stable across installs.
    device_kwargs: dict[str, object] = {}
    training_args_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    if args.device == "cpu":
        if "no_cuda" in training_args_params:
            device_kwargs["no_cuda"] = True
        elif "use_cpu" in training_args_params:
            device_kwargs["use_cpu"] = True
        else:
            # Fallback: Trainer will likely still choose CUDA if available, but we at least warn.
            print("Warning: this Transformers version has no 'no_cuda'/'use_cpu' TrainingArguments flag; cannot force CPU.")

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        do_train=True,
        do_eval=("validation" in lm_ds and (not args.no_eval)),
        num_train_epochs=float(args.epochs),
        learning_rate=float(args.lr),
        per_device_train_batch_size=int(args.batch_size),
        per_device_eval_batch_size=int(args.eval_batch_size),
        gradient_accumulation_steps=int(args.grad_accum),
        warmup_ratio=float(args.warmup_ratio),
        weight_decay=float(args.weight_decay),
        logging_steps=int(args.logging_steps),
        save_steps=int(args.save_steps),
        save_total_limit=2,
        eval_strategy=("steps" if ("validation" in lm_ds and (not args.no_eval)) else "no"),
        eval_steps=int(args.save_steps),
        report_to=[],
        dataloader_num_workers=int(args.num_workers),
        fp16=effective_fp16,
        bf16=bool(args.bf16),
        **device_kwargs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_ds["train"],
        eval_dataset=(lm_ds["validation"] if ("validation" in lm_ds and (not args.no_eval)) else None),
        processing_class=tokenizer,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))

    print(f"Saved fine-tuned model to: {out_dir}")


if __name__ == "__main__":
    main()

