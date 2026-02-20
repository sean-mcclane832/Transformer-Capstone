from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from model.gpt2 import GPT2Config, GPT2LMHeadModel
from tokenizer.bpe import UTF8BPE


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate text from NumPy GPT-2 checkpoint")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_model(ckpt_dir: Path) -> tuple[GPT2LMHeadModel, UTF8BPE]:
    meta = json.loads((ckpt_dir / "meta.json").read_text(encoding="utf-8"))
    config = GPT2Config(**meta["config"])
    model = GPT2LMHeadModel(config)

    weights = np.load(ckpt_dir / "weights.npz")
    params = model.parameters()
    expected = {f"p_{i}" for i in range(len(params))}
    found = set(weights.files)
    missing = sorted(expected - found)
    if missing:
        raise ValueError(f"checkpoint is missing parameter arrays: {missing[:5]}")
    for i, p in enumerate(params):
        p.data[...] = weights[f"p_{i}"]

    tokenizer = UTF8BPE.load(ckpt_dir / meta["tokenizer_file"])
    return model, tokenizer


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    model, tokenizer = load_model(args.checkpoint_dir)
    ids = tokenizer.encode(args.prompt, add_bos=True)
    x = np.array([ids], dtype=np.int64)

    y = model.generate(x, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(tokenizer.decode(y[0].tolist()))


if __name__ == "__main__":
    main()
