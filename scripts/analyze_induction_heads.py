"""
analyze_induction_heads.py

Detects and visualises induction heads in ADRIA.

Induction heads (Olsson et al. 2022) implement a two-step in-context learning
primitive: if the context contains ...A B ... A, the head attends strongly to
position B, predicting B will follow the second A.

Detection method (Olsson et al., §3):
1. Build a repeated-sequence input: [T1 T2 ... T_N T1 T2 ... T_N]
   (same N-token sequence twice)
2. For every attention head, compute the mean attention weight on the
   *induction stripe*: position (i, i - N + 1) for i >= N.
   High score → head attends "one copy ago", i.e. induction behaviour.
3. Visualise the per-head scores as a heatmap (layer × head) and plot the
   attention pattern of the top-scoring head.

Usage:
    python scripts/analyze_induction_heads.py --checkpoint checkpoints/best.pt
    python scripts/analyze_induction_heads.py --checkpoint checkpoints/best.pt --seq-len 64
"""

import sys
import argparse
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model.gpt import GPT
from utils.config import GENERAL_CONFIG
import utils.config as _cfg


def load_model(ckpt_path: Path, device: torch.device) -> GPT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "gen_config" in ckpt:
        original = dict(_cfg.GENERAL_CONFIG)
        _cfg.GENERAL_CONFIG.clear()
        _cfg.GENERAL_CONFIG.update(ckpt["gen_config"])
    # Force attention weights on for this analysis
    _cfg.GENERAL_CONFIG["return_attn_weights"] = True
    m = GPT().to(device)
    if "gen_config" in ckpt:
        _cfg.GENERAL_CONFIG.clear()
        _cfg.GENERAL_CONFIG.update(original)
        _cfg.GENERAL_CONFIG["return_attn_weights"] = True
    state = ckpt.get("model", ckpt)
    m.load_state_dict(state)
    m.eval()
    return m


def collect_attention_weights(model: GPT, idx: torch.Tensor, device: torch.device):
    """Run one forward pass and collect per-layer attention weight matrices."""
    # Hook into each block's attention module
    all_weights = []

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is (attn_out, weights) when return_attn_weights=True
            if isinstance(output, tuple):
                all_weights.append(output[1].detach().cpu())
        return hook

    handles = []
    for i, block in enumerate(model.blocks):
        h = block.attention.register_forward_hook(make_hook(i))
        handles.append(h)

    with torch.no_grad():
        model(idx.to(device))

    for h in handles:
        h.remove()

    return all_weights  # list of (1, n_heads, T, T) tensors


def induction_score(weights: torch.Tensor, prefix_len: int) -> torch.Tensor:
    """
    Given attention weights (1, n_heads, T, T) and prefix_len N,
    return per-head induction score: mean weight on the diagonal offset -(N-1).

    The "induction stripe" is where token i attends to token i - (N-1):
    for i >= N, position [i, i - (N-1)] is where token i would attend if
    it recognised that the same token appeared N positions ago.
    """
    _, n_heads, T, _ = weights.shape
    scores = torch.zeros(n_heads)
    offset = -(prefix_len - 1)   # negative = below main diagonal

    for h in range(n_heads):
        w = weights[0, h]  # (T, T)
        # Extract the diagonal at the given offset
        diag = torch.diagonal(w, offset=offset)  # length T - |offset|
        # Only count positions where the full repeated window applies
        if diag.numel() > 0:
            scores[h] = diag[prefix_len:].mean().item() if diag.numel() > prefix_len else diag.mean().item()

    return scores


def build_repeated_sequence(seq_len: int, vocab_size: int, seed: int = 42) -> torch.Tensor:
    rng = random.Random(seed)
    half = seq_len // 2
    # Avoid BOS/EOS tokens (IDs 0 and 1 by convention)
    tokens = [rng.randint(2, vocab_size - 1) for _ in range(half)]
    repeated = tokens + tokens
    return torch.tensor([repeated], dtype=torch.long)


def main():
    parser = argparse.ArgumentParser(description="Induction head analysis for ADRIA")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--seq-len",    type=int, default=128,
                        help="Length of repeated sequence (will be halved for each copy)")
    parser.add_argument("--device",     type=str, default=None)
    parser.add_argument("--out-dir",    type=str, default="figures")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(exist_ok=True)

    print(f"Loading model from {ckpt_path} ...")
    model = load_model(ckpt_path, device)

    n_layers = len(model.blocks)
    n_heads  = GENERAL_CONFIG["n_heads"]
    seq_len  = min(args.seq_len, GENERAL_CONFIG["max_seq_len"])
    prefix_len = seq_len // 2

    print(f"  {n_layers} layers x {n_heads} heads | sequence length {seq_len} (prefix={prefix_len})")

    # Average induction scores across multiple random sequences for stability
    n_samples = 10
    all_scores = torch.zeros(n_layers, n_heads)

    for s in range(n_samples):
        idx = build_repeated_sequence(seq_len, GENERAL_CONFIG["vocab_size"], seed=s)
        layer_weights = collect_attention_weights(model, idx, device)

        for layer_idx, w in enumerate(layer_weights):
            scores = induction_score(w, prefix_len)
            all_scores[layer_idx] += scores

    all_scores /= n_samples

    # ── Heatmap: layer x head induction scores ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Induction Head Analysis — ADRIA", fontsize=14)

    ax = axes[0]
    im = ax.imshow(all_scores.numpy(), aspect="auto", cmap="viridis")
    ax.set_xlabel("Head")
    ax.set_ylabel("Layer")
    ax.set_title("Per-head induction score\n(mean attention on repeated-sequence diagonal)")
    ax.set_xticks(range(n_heads))
    ax.set_yticks(range(n_layers))
    plt.colorbar(im, ax=ax)

    # ── Attention pattern of the top-scoring head ─────────────────────────────
    best_layer, best_head = np.unravel_index(all_scores.numpy().argmax(), all_scores.shape)
    print(f"\nTop induction head: layer {best_layer}, head {best_head} "
          f"(score={all_scores[best_layer, best_head]:.3f})")

    # Collect weights for one example
    idx_example = build_repeated_sequence(seq_len, GENERAL_CONFIG["vocab_size"], seed=99)
    layer_weights_ex = collect_attention_weights(model, idx_example, device)
    top_w = layer_weights_ex[best_layer][0, best_head].numpy()   # (T, T)

    ax2 = axes[1]
    im2 = ax2.imshow(top_w, aspect="auto", cmap="Blues", vmin=0)
    ax2.set_title(f"Attention pattern: layer {best_layer} head {best_head}\n"
                  f"(highest induction score={all_scores[best_layer, best_head]:.3f})")
    ax2.set_xlabel("Key position (attended to)")
    ax2.set_ylabel("Query position")
    ax2.axvline(x=prefix_len - 0.5, color="red", linewidth=1, linestyle="--", label=f"repeat boundary (pos {prefix_len})")
    ax2.legend(fontsize=7)
    plt.colorbar(im2, ax=ax2)

    plt.tight_layout()
    out_path = out_dir / "induction_heads.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved {out_path}")

    # Print top 5 induction heads
    flat_scores = [(all_scores[l, h].item(), l, h)
                   for l in range(n_layers) for h in range(n_heads)]
    flat_scores.sort(reverse=True)
    print("\nTop 5 induction heads:")
    print(f"  {'Score':>7}  {'Layer':>5}  {'Head':>4}")
    for score, l, h in flat_scores[:5]:
        print(f"  {score:7.4f}  {l:5d}  {h:4d}")

    # ── Per-layer mean induction score (bar chart) ────────────────────────────
    fig2, ax3 = plt.subplots(figsize=(8, 4))
    layer_means = all_scores.mean(dim=1).numpy()
    ax3.bar(range(n_layers), layer_means, color="steelblue")
    ax3.set_xlabel("Layer")
    ax3.set_ylabel("Mean induction score")
    ax3.set_title("Mean induction score per layer — ADRIA")
    ax3.set_xticks(range(n_layers))
    plt.tight_layout()
    out_path2 = out_dir / "induction_heads_by_layer.png"
    plt.savefig(out_path2, dpi=150)
    plt.close()
    print(f"Saved {out_path2}")


if __name__ == "__main__":
    main()
