"""
plot_curves.py — Training curve visualizations for ADRIA

Single-run mode (default):
    python scripts/plot_curves.py --log figures/run_log.pt
    python scripts/plot_curves.py --log figures/runs/small-rope/run_log.pt --out-dir figures/runs/small-rope
    python scripts/plot_curves.py --log figures/run_log.pt --checkpoint checkpoints/best.pt --prompt "To be"

Comparison mode (overlay multiple runs on one chart):
    python scripts/plot_curves.py --compare figures/runs/small-base/run_log.pt figures/runs/small-rope/run_log.pt
    python scripts/plot_curves.py --compare figures/runs/*/run_log.pt --out-dir figures/comparison
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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

FIGURES_DIR = ROOT / "figures"

# Colour cycle for comparison plots
PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
           "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac"]


# ── Single-run plots ──────────────────────────────────────────────────────────

def plot_loss(train_steps, train_losses, val_steps, val_losses, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(train_steps, train_losses, label="train", linewidth=1)
    axes[0].plot(val_steps,   val_losses,   label="val",   linewidth=1.5)
    axes[0].set_xlabel("step")
    axes[0].set_ylabel("loss (nats)")
    axes[0].set_title("Cross-Entropy Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

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


def plot_bpc(val_steps, val_losses, out_path: Path) -> None:
    bpt = [l / math.log(2) for l in val_losses]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(val_steps, bpt, linewidth=1.5, label="ADRIA (val)")
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


def plot_layer_grad_norms(layer_gnorm_steps, layer_gnorms, out_path: Path) -> None:
    data     = np.array(layer_gnorms)
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


def plot_attention_heatmaps(weights: torch.Tensor, tokens: list[str], out_path: Path) -> None:
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


# ── Comparison plots (multiple runs on one chart) ─────────────────────────────

def _run_label(log_path: str) -> str:
    """Derive a short label from the log file path."""
    p = Path(log_path)
    # figures/runs/small-rope/run_log.pt → "small-rope"
    if p.parent.name != "figures" and p.parent.name != ".":
        return p.parent.name
    return p.stem


def plot_compare_loss(run_logs: list[tuple[str, dict]], out_path: Path) -> None:
    """
    Overlay validation loss curves from multiple runs.
    run_logs: list of (label, log_dict)
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Run Comparison — Val Loss & Perplexity", fontsize=13)

    for i, (label, log) in enumerate(run_logs):
        c = PALETTE[i % len(PALETTE)]
        val_steps  = log["val_steps"]
        val_losses = log["val_losses"]
        val_ppl    = [math.exp(min(l, 20)) for l in val_losses]

        axes[0].plot(val_steps, val_losses, label=label, color=c, linewidth=1.8)
        axes[1].plot(val_steps, val_ppl,    label=label, color=c, linewidth=1.8)

    for ax, ylabel, title in zip(
        axes,
        ["val loss (nats)", "perplexity"],
        ["Validation Loss", "Validation Perplexity"],
    ):
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(alpha=0.3)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_compare_tokens_loss(run_logs: list[tuple[str, dict]], out_path: Path) -> None:
    """Overlay val loss vs tokens seen — equalises runs with different step counts."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Val Loss vs Tokens Seen — Run Comparison", fontsize=12)

    for i, (label, log) in enumerate(run_logs):
        c = PALETTE[i % len(PALETTE)]
        if not (log.get("tokens_seen") and log.get("val_steps")):
            continue
        step_to_tokens = {s: t for s, t in zip(log["train_steps"], log["tokens_seen"])}
        val_steps  = log["val_steps"]
        val_losses = log["val_losses"]
        val_tokens = [step_to_tokens[s] for s in val_steps if s in step_to_tokens]
        losses     = [val_losses[j] for j, s in enumerate(val_steps) if s in step_to_tokens]
        if val_tokens:
            ax.plot(val_tokens, losses, label=label, color=c, linewidth=1.8)

    ax.set_xlabel("tokens seen")
    ax.set_ylabel("val loss (nats)")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x/1e6:.1f}M"))
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


def plot_compare_bpc(run_logs: list[tuple[str, dict]], out_path: Path) -> None:
    """Overlay bits-per-token curves with GPT-2 reference lines."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Bits per Token — Run Comparison", fontsize=12)

    for i, (label, log) in enumerate(run_logs):
        c = PALETTE[i % len(PALETTE)]
        bpt = [l / math.log(2) for l in log["val_losses"]]
        ax.plot(log["val_steps"], bpt, label=label, color=c, linewidth=1.8)

    ax.axhline(math.log2(29.6), color="gray",  linestyle="--", linewidth=1, label="GPT-2 Small")
    ax.axhline(math.log2(22.8), color="silver", linestyle="--", linewidth=1, label="GPT-2 Medium")
    ax.set_xlabel("step")
    ax.set_ylabel("bits per token")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ADRIA training curves")

    # Single-run mode
    parser.add_argument("--log",        type=str, default=None,
                        help="Path to a single training log .pt file")
    parser.add_argument("--out-dir",    type=str, default=None,
                        help="Output directory for figures (default: same dir as --log, "
                             "or figures/ if not specified)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Optional checkpoint for attention heatmap")
    parser.add_argument("--prompt",     type=str, default=None,
                        help="Prompt text for attention heatmap (requires --checkpoint)")

    # Comparison mode
    parser.add_argument("--compare",    type=str, nargs="+", default=None,
                        help="Two or more run_log.pt paths to overlay in comparison plots")
    parser.add_argument("--labels",     type=str, nargs="+", default=None,
                        help="Legend labels for --compare (must match count); "
                             "defaults to parent directory name of each log")

    args = parser.parse_args()

    # ── Comparison mode ───────────────────────────────────────────────────────
    if args.compare:
        out_dir = Path(args.out_dir) if args.out_dir else FIGURES_DIR / "comparison"
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        labels = args.labels or [_run_label(p) for p in args.compare]
        if len(labels) != len(args.compare):
            parser.error("--labels count must match --compare count")

        run_logs = []
        for label, path in zip(labels, args.compare):
            p = Path(path)
            if not p.is_absolute():
                p = ROOT / p
            log = torch.load(p, weights_only=False)
            run_logs.append((label, log))
            min_val = min(log["val_losses"])
            print(f"  {label}: {len(log['val_steps'])} val points, "
                  f"best val loss {min_val:.4f} (ppl {math.exp(min(min_val, 20)):.1f})")

        plot_compare_loss(run_logs, out_dir / "compare_loss.png")
        plot_compare_tokens_loss(run_logs, out_dir / "compare_tokens_loss.png")
        plot_compare_bpc(run_logs, out_dir / "compare_bpc.png")
        return

    # ── Single-run mode ───────────────────────────────────────────────────────
    if args.log is None:
        parser.error("Provide --log <path> for single-run mode, or --compare <paths...> for comparison")

    log_path = Path(args.log)
    if not log_path.is_absolute():
        log_path = ROOT / log_path

    if args.out_dir:
        out_dir = Path(args.out_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
    else:
        out_dir = log_path.parent if log_path.parent.name != "figures" else FIGURES_DIR

    out_dir.mkdir(parents=True, exist_ok=True)
    log = torch.load(log_path, weights_only=False)

    plot_loss(
        log["train_steps"], log["train_losses"],
        log["val_steps"],   log["val_losses"],
        out_dir / "loss_perplexity.png",
    )
    plot_lr_schedule(log["train_steps"], log["lrs"], out_dir / "lr_schedule.png")

    if log.get("gnorms"):
        plot_grad_norm(log["train_steps"], log["gnorms"], out_dir / "grad_norm.png")

    if log.get("val_losses"):
        plot_bpc(log["val_steps"], log["val_losses"], out_dir / "bpc.png")

    if log.get("tokens_seen") and log.get("val_steps"):
        step_to_tokens = {s: t for s, t in zip(log["train_steps"], log["tokens_seen"])}
        plot_tokens_vs_loss(log["val_steps"], log["val_losses"], step_to_tokens,
                            out_dir / "tokens_vs_loss.png")

    if log.get("layer_gnorms"):
        plot_layer_grad_norms(log["layer_gnorm_steps"], log["layer_gnorms"],
                              out_dir / "layer_grad_norms.png")

    if log.get("pos_losses"):
        plot_loss_by_position(log["pos_losses"], out_dir / "loss_by_position.png")

    if args.checkpoint and args.prompt:
        import utils.config as _cfg
        from model.gpt import GPT
        from text_processing.token_class import ByteBPETokenizer
        from utils.config import TOKENIZER_CONFIG

        tokenizer = ByteBPETokenizer.load(TOKENIZER_CONFIG["output"])

        ckpt_path = Path(args.checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = ROOT / ckpt_path
        ckpt   = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        device = torch.device(_cfg.GENERAL_CONFIG["device"])

        if "gen_config" in ckpt:
            original = dict(_cfg.GENERAL_CONFIG)
            _cfg.GENERAL_CONFIG.clear()
            _cfg.GENERAL_CONFIG.update(ckpt["gen_config"])
        _cfg.GENERAL_CONFIG["return_attn_weights"] = True
        model = GPT().to(device)
        if "gen_config" in ckpt:
            _cfg.GENERAL_CONFIG.clear()
            _cfg.GENERAL_CONFIG.update(original)
            _cfg.GENERAL_CONFIG["return_attn_weights"] = True
        model.load_state_dict(ckpt.get("model", ckpt))
        model.eval()

        ids = tokenizer.encode(args.prompt, add_bos=False, add_eos=False)
        idx = torch.tensor([ids], dtype=torch.long, device=device)

        captured = {}
        last_attn = model.blocks[-1].attention
        handle = last_attn.register_forward_hook(
            lambda _m, _inp, out: captured.update({"weights": out[1].detach().cpu()})
        )
        with torch.no_grad():
            model(idx)
        handle.remove()
        _cfg.GENERAL_CONFIG["return_attn_weights"] = False

        if "weights" in captured:
            weights = captured["weights"][0]
            token_strs = [tokenizer.decode([t]) for t in ids]
            plot_attention_heatmaps(weights, token_strs, out_dir / "attention_heatmap.png")
        else:
            print("Warning: attention weights not captured")


if __name__ == "__main__":
    main()
