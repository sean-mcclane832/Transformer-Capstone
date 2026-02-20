from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from tqdm import tqdm

from model.gpt2 import AdamW, GPT2Config, GPT2LMHeadModel, GPT2_PRESETS
from tokenizer.bpe import UTF8BPE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train GPT-2 style LM in pure NumPy")
    p.add_argument("--train-text", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("artifacts"))
    p.add_argument("--model-size", choices=list(GPT2_PRESETS.keys()), default="gpt2")
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument("--block-size", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def build_batch(tokens: np.ndarray, block_size: int, batch_size: int, rng: np.random.Generator):
    max_start = len(tokens) - block_size - 1
    starts = rng.integers(0, max_start + 1, size=batch_size)
    x = np.stack([tokens[s : s + block_size] for s in starts], axis=0)
    y = np.stack([tokens[s + 1 : s + block_size + 1] for s in starts], axis=0)
    return x.astype(np.int64), y.astype(np.int64)


def save_checkpoint(model: GPT2LMHeadModel, tokenizer_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for i, p in enumerate(model.parameters()):
        arrays[f"p_{i}"] = p.data
    np.savez(out_dir / "weights.npz", **arrays)

    cfg = model.config.__dict__.copy()
    payload = {"config": cfg, "tokenizer_file": tokenizer_path.name}
    (out_dir / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    text = args.train_text.read_text(encoding="utf-8")
    tokenizer = UTF8BPE(vocab_size=args.vocab_size)
    print("Training UTF-8 BPE tokenizer...")
    tokenizer.train(text)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_path = args.out_dir / "tokenizer.json"
    tokenizer.save(tokenizer_path)

    tokens = np.array(tokenizer.encode(text, add_bos=True, add_eos=True), dtype=np.int64)
    if len(tokens) < args.block_size + 2:
        raise ValueError("Training corpus is too short for the selected block size")

    base = GPT2_PRESETS[args.model_size]
    config = GPT2Config(
        vocab_size=args.vocab_size,
        block_size=args.block_size,
        n_layer=base.n_layer,
        n_head=base.n_head,
        n_embd=base.n_embd,
        bias=base.bias,
    )

    model = GPT2LMHeadModel(config)
    opt = AdamW(model.parameters(), lr=args.lr)
    print(f"Model parameters: {model.num_parameters():,}")

    for step in tqdm(range(1, args.steps + 1), desc="training"):
        x, y = build_batch(tokens, args.block_size, args.batch_size, rng)
        model.zero_grad()
        loss = model.loss_and_backward(x, y)
        opt.step()
        if step % args.log_every == 0:
            print(f"step {step}: loss={loss:.4f}")

    save_checkpoint(model, tokenizer_path, args.out_dir)
    print(f"Saved checkpoint to: {args.out_dir}")


if __name__ == "__main__":
    main()
