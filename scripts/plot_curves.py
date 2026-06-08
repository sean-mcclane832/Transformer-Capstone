"""
plot_curves.py — Training curve visualizations for ADRIA

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
import numpy as np

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


# ── Gradient norm ─────────────────────────────────────────────────────────────

def plot_grad_norm(steps, gnorms, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(steps, gnorms, linewidth=0.8, alpha=0.7, color="steelblue", label="grad norm")
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="clip threshold (1.0)")
    ax.set_xlabel("step")
    ax.set_ylabel("gradient norm (pre-clip)")
    ax.set_title("Gradient Norm over Training")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Bits per token ─────────────────────────────────────────────────────────────

def plot_bpc(val_steps, val_losses, out_path: Path) -> None:
    bpt = [l / math.log(2) for l in val_losses]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(val_steps, bpt, linewidth=1.5, label="ADRIA (val)")
    # GPT-2 reference lines on WikiText-103 (bits per token, 50K vocab BPE)
    ax.axhline(math.log2(29.6), color="gray",  linestyle="--", linewidth=1, label="GPT-2 Small  (~29.6 ppl)")
    ax.axhline(math.log2(22.8), color="silver", linestyle="--", linewidth=1, label="GPT-2 Medium (~22.8 ppl)")
    ax.set_xlabel("step")
    ax.set_ylabel("bits per token")
    ax.set_title("Bits per Token — Val Set")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Tokens seen vs val loss ───────────────────────────────────────────────────

def plot_tokens_vs_loss(val_steps, val_losses, step_to_tokens, out_path: Path) -> None:
    val_tokens = [step_to_tokens[s] for s in val_steps if s in step_to_tokens]
    losses     = [val_losses[i] for i, s in enumerate(val_steps) if s in step_to_tokens]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(val_tokens, losses, linewidth=1.5, marker="o", markersize=3)
    ax.set_xlabel("tokens seen")
    ax.set_ylabel("val loss (nats)")
    ax.set_title("Val Loss vs Tokens Seen")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Per-layer gradient norms ──────────────────────────────────────────────────

def plot_layer_grad_norms(layer_gnorm_steps, layer_gnorms, out_path: Path) -> None:
    data     = np.array(layer_gnorms)   # (n_log_steps, n_layers)
    n_layers = data.shape[1]
    fig, ax  = plt.subplots(figsize=(10, 4))
    for i in range(n_layers):
        ax.plot(layer_gnorm_steps, data[:, i], linewidth=1.2, label=f"Layer {i}")
    ax.set_xlabel("step")
    ax.set_ylabel("gradient norm")
    ax.set_title("Per-Layer Gradient Norms (pre-clip, every 50 steps)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Loss by sequence position ─────────────────────────────────────────────────

def plot_loss_by_position(pos_losses, out_path: Path) -> None:
    positions = list(range(len(pos_losses)))
    fig, ax   = plt.subplots(figsize=(10, 4))
    ax.plot(positions, pos_losses, linewidth=1.5, color="steelblue")
    ax.set_xlabel("sequence position")
    ax.set_ylabel("average loss (nats)")
    ax.set_title("Loss by Sequence Position (val set)\nEarly positions have less context — loss should decrease with position")
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
    parser = argparse.ArgumentParser(description="Plot ADRIA training curves")
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

    if log.get("gnorms"):
        plot_grad_norm(log["train_steps"], log["gnorms"], FIGURES_DIR / "grad_norm.png")

    if log.get("val_losses"):
        plot_bpc(log["val_steps"], log["val_losses"], FIGURES_DIR / "bpc.png")

    if log.get("tokens_seen") and log.get("val_steps"):
        step_to_tokens = {s: t for s, t in zip(log["train_steps"], log["tokens_seen"])}
        plot_tokens_vs_loss(log["val_steps"], log["val_losses"], step_to_tokens, FIGURES_DIR / "tokens_vs_loss.png")

    if log.get("layer_gnorms"):
        plot_layer_grad_norms(log["layer_gnorm_steps"], log["layer_gnorms"], FIGURES_DIR / "layer_grad_norms.png")

    if log.get("pos_losses"):
        plot_loss_by_position(log["pos_losses"], FIGURES_DIR / "loss_by_position.png")

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

        # Capture attention weights from the last block via a forward hook —
        # avoids changing GPT's interface; weights are (batch, n_heads, seq, seq)
        captured = {}
        last_attn = model.blocks[-1].attention
        last_attn.return_attn_weights = True
        handle = last_attn.register_forward_hook(
            lambda _m, _inp, out: captured.update({"weights": out[1].detach().cpu()})
        )
        with torch.no_grad():
            model(idx)
        handle.remove()
        last_attn.return_attn_weights = False  # type: ignore[assignment]

        if "weights" in captured:
            weights = captured["weights"][0]   # (n_heads, seq_len, seq_len) — first batch item
            token_strs = [tokenizer.decode([t]) for t in ids]
            plot_attention_heatmaps(weights, token_strs, FIGURES_DIR / "attention_heatmap.png")
        else:
            print("Warning: attention weights not captured")


if __name__ == "__main__":
    main()
