"""
analyze_domain_loss.py

Evaluates ADRIA's perplexity separately on each corpus domain to understand
what kinds of text the model has learned to model better than others.

Domains:
  - Shakespeare (input.txt)        — Early Modern English, dense poetry/drama
  - The Great Gatsby (greatgatsby.txt) — 1920s prose fiction
  - WikiText-103 (wikitext103.txt) — Wikipedia articles, factual prose
  - PG-19 (pg19.txt)               — 19th-century Gutenberg novels
  - OpenWebText (openwebtext.txt)  — Modern web text

The model was trained on a mix of all five, so this tells us which domain
distribution it fit most/least tightly given its capacity.

Usage:
    python scripts/analyze_domain_loss.py --checkpoint checkpoints/best.pt
    python scripts/analyze_domain_loss.py --checkpoint checkpoints/best.pt --max-tokens 50000
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import math
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model.gpt import GPT
from text_processing.token_class import ByteBPETokenizer
from utils.config import GENERAL_CONFIG, TOKENIZER_CONFIG
import utils.config as _cfg


DOMAIN_FILES = {
    "Shakespeare":    ROOT / "data" / "raw" / "input.txt",
    "Great Gatsby":   ROOT / "data" / "raw" / "greatgatsby.txt",
    "WikiText-103":   ROOT / "data" / "raw" / "wikitext103.txt",
    "PG-19":          ROOT / "data" / "raw" / "pg19.txt",
    "OpenWebText":    ROOT / "data" / "raw" / "openwebtext.txt",
}


def load_model(ckpt_path: Path, device: torch.device) -> GPT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "gen_config" in ckpt:
        original = dict(_cfg.GENERAL_CONFIG)
        _cfg.GENERAL_CONFIG.update(ckpt["gen_config"])
    m = GPT().to(device)
    if "gen_config" in ckpt:
        _cfg.GENERAL_CONFIG.clear()
        _cfg.GENERAL_CONFIG.update(original)
    state = ckpt.get("model", ckpt)
    m.load_state_dict(state)
    m.eval()
    return m


@torch.no_grad()
def evaluate_tokens(model: GPT, token_ids: torch.Tensor, device: torch.device,
                    seq_len: int, batch_size: int = 8) -> tuple[float, int]:
    """Evaluate cross-entropy loss over a flat token sequence."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n = len(token_ids)

    for start in range(0, n - seq_len, seq_len * batch_size):
        batch_xs, batch_ys = [], []
        for b in range(batch_size):
            s = start + b * seq_len
            if s + seq_len >= n:
                break
            x = token_ids[s : s + seq_len]
            y = token_ids[s + 1 : s + seq_len + 1]
            batch_xs.append(x)
            batch_ys.append(y)

        if not batch_xs:
            break

        x_batch = torch.stack(batch_xs).to(device)  # (B, seq_len)
        y_batch = torch.stack(batch_ys).to(device)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda", dtype=torch.float16):
            logits, _ = model(x_batch)

        B, T, V = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V), y_batch.view(B * T), reduction="sum")
        total_loss += loss.item()
        total_tokens += B * T

    model.train()
    return total_loss / max(total_tokens, 1), total_tokens


def main():
    parser = argparse.ArgumentParser(description="Per-domain loss analysis for ADRIA")
    parser.add_argument("--checkpoint",  type=str, default="checkpoints/best.pt")
    parser.add_argument("--max-tokens",  type=int, default=100_000,
                        help="Max tokens to evaluate per domain (default: 100000)")
    parser.add_argument("--device",      type=str, default=None)
    parser.add_argument("--out-dir",     type=str, default="figures")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(exist_ok=True)

    print(f"Loading tokenizer ...")
    tokenizer = ByteBPETokenizer.load(TOKENIZER_CONFIG["output"])

    print(f"Loading model from {ckpt_path} on {device} ...")
    model = load_model(ckpt_path, device)
    seq_len = GENERAL_CONFIG["max_seq_len"]

    results = {}
    for domain, path in DOMAIN_FILES.items():
        if not path.exists():
            print(f"  [{domain}] file not found, skipping")
            continue

        print(f"  [{domain}] tokenizing {path.name} ...")
        text = path.read_text(encoding="utf-8", errors="replace")
        # Use the first max_tokens worth of characters (rough estimate: avg 4 chars/token)
        char_limit = args.max_tokens * 4
        if len(text) > char_limit:
            text = text[:char_limit]

        ids = tokenizer.encode(text, add_bos=False, add_eos=False)
        ids_t = torch.tensor(ids[:args.max_tokens], dtype=torch.long)

        if len(ids_t) < seq_len + 1:
            print(f"  [{domain}] too short ({len(ids_t)} tokens), skipping")
            continue

        loss, n_tokens = evaluate_tokens(model, ids_t, device, seq_len)
        ppl = math.exp(min(loss, 20))
        results[domain] = {"loss": loss, "ppl": ppl, "n_tokens": n_tokens}
        print(f"  [{domain}] loss={loss:.4f}  ppl={ppl:.2f}  ({n_tokens:,} tokens evaluated)")

    if not results:
        print("No domains evaluated.")
        return

    # ── Bar chart ─────────────────────────────────────────────────────────────
    domains = list(results.keys())
    ppls    = [results[d]["ppl"]  for d in domains]
    losses  = [results[d]["loss"] for d in domains]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Per-domain language model evaluation — ADRIA", fontsize=13)

    colours = plt.cm.tab10(range(len(domains)))

    axes[0].bar(domains, losses, color=colours)
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].set_title("Cross-entropy by domain")
    axes[0].tick_params(axis="x", rotation=20)
    for i, v in enumerate(losses):
        axes[0].text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)

    axes[1].bar(domains, ppls, color=colours)
    axes[1].set_ylabel("Perplexity")
    axes[1].set_title("Perplexity by domain")
    axes[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(ppls):
        axes[1].text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "domain_loss.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nSaved {out_path}")

    # Summary table
    print("\nDomain summary:")
    print(f"  {'Domain':<20}  {'Loss':>8}  {'PPL':>8}  {'Tokens':>10}")
    print("  " + "-" * 54)
    for d in domains:
        r = results[d]
        print(f"  {d:<20}  {r['loss']:8.4f}  {r['ppl']:8.2f}  {r['n_tokens']:10,}")


if __name__ == "__main__":
    main()
