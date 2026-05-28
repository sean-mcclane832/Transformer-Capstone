"""
plot_curves.py — Training curve visualizations for MiniGPT

Reads a checkpoint and/or a log file, then produces:
  - Loss curve (train + val)
  - Perplexity curve (train + val)
  - Learning rate schedule
  - (Optional) per-head attention heatmaps for a sample prompt

Usage:
    python scripts/plot_curves.py --log figures/run_log.pt
    python scripts/plot_curves.py --log figures/run_log.pt --checkpoint checkpoints/best.pt --prompt "To be"
"""

import sys
import argparse
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FIGURES_DIR = ROOT / "figures"


# ── Loss & perplexity curves ──────────────────────────────────────────────────

def plot_loss(train_steps, train_losses, val_steps, val_losses, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(train_steps, train_losses, label="train", linewidth=1)
    axes[0].plot(val_steps,   val_losses,   label="val",   linewidth=1.5)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss (nats)")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Perplexity
    train_ppl = [math.exp(min(l, 20)) for l in train_losses]
    val_ppl   = [math.exp(min(l, 20)) for l in val_losses]
    axes[1].plot(train_steps, train_ppl, label="train", linewidth=1)
    axes[1].plot(val_steps,   val_ppl,   label="val",   linewidth=1.5)
    axes[1].set_xlabel("step")
    axes[1].set_ylabel("perplexity")
    axes[1].set_title("Perplexity")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── LR schedule curve ─────────────────────────────────────────────────────────

def plot_lr_schedule(steps, lrs, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(steps, lrs, linewidth=1.5)
    ax.set_xlabel("step")
    ax.set_ylabel("learning rate")
    ax.set_title("LR Schedule (warmup + cosine decay)")
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2e"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Attention heatmaps ────────────────────────────────────────────────────────

def plot_attention_heatmaps(weights: torch.Tensor, tokens: list[str], out_path: Path) -> None:
    """
    weights: (n_heads, seq_len, seq_len) attention weight tensor
    tokens:  list of string tokens for axis labels
    """
    n_heads = weights.size(0)
    fig, axes = plt.subplots(1, n_heads, figsize=(4 * n_heads, 4))
    if n_heads == 1:
        axes = [axes]

    for h, ax in enumerate(axes):
        im = ax.imshow(weights[h].cpu().numpy(), vmin=0, vmax=1, cmap="Blues")
        ax.set_title(f"Head {h}")
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Attention Weights (last block)", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot MiniGPT training curves")
    parser.add_argument("--log",        type=str, required=True,
                        help="Path to training log .pt file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional checkpoint for attention heatmap")
    parser.add_argument("--prompt",     type=str, default=None,
                        help="Prompt text for attention heatmap (requires --checkpoint)")
    args = parser.parse_args()

    log = torch.load(args.log, weights_only=False)

    plot_loss(
        log["train_steps"], log["train_losses"],
        log["val_steps"],   log["val_losses"],
        FIGURES_DIR / "loss_perplexity.png",
    )
    plot_lr_schedule(
        log["train_steps"], log["lrs"],
        FIGURES_DIR / "lr_schedule.png",
    )

    if args.checkpoint and args.prompt:
        from model.gpt import GPT
        from text_processing.token_class import ByteBPETokenizer
        from utils.config import TOKENIZER_CONFIG, GENERAL_CONFIG

        tokenizer = ByteBPETokenizer()
        tokenizer.load(TOKENIZER_CONFIG["output"])

        device = torch.device(GENERAL_CONFIG["device"])
        ckpt   = torch.load(args.checkpoint, map_location=device)
        model  = GPT().to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        ids = tokenizer.encode(args.prompt, add_bos=False, add_eos=False)
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        # Run forward with return_attn_weights=True (last block's weights)
        from utils.config import GENERAL_CONFIG as cfg
        # TODO: collect attention weights from the last block's forward pass
        print("Attention heatmap generation: TODO — requires hooking into block forward")


if __name__ == "__main__":
    main()
