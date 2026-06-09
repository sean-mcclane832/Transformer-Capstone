"""
analyze_embeddings.py

Visualises the learned token embedding space of ADRIA via PCA.

Two plots are produced:
1. PCA scatter of all 32,768 token embeddings coloured by token frequency
   (embeddings of common tokens cluster differently from rare ones)
2. Annotated scatter of the top-N most frequent tokens, so readable words
   appear in the 2D projection and we can see if semantically related tokens
   cluster together.

Usage:
    python scripts/analyze_embeddings.py --checkpoint checkpoints/best.pt
    python scripts/analyze_embeddings.py --checkpoint checkpoints/best.pt --top-n 300
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from model.gpt import GPT
from text_processing.token_class import ByteBPETokenizer
from utils.config import GENERAL_CONFIG, TOKENIZER_CONFIG
import utils.config as _cfg


def load_model(ckpt_path: Path, device: torch.device) -> GPT:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if "gen_config" in ckpt:
        original = dict(_cfg.GENERAL_CONFIG)
        _cfg.GENERAL_CONFIG.clear()
        _cfg.GENERAL_CONFIG.update(ckpt["gen_config"])
    m = GPT().to(device)
    if "gen_config" in ckpt:
        _cfg.GENERAL_CONFIG.clear()
        _cfg.GENERAL_CONFIG.update(original)
    state = ckpt.get("model", ckpt)
    m.load_state_dict(state)
    m.eval()
    return m


def main():
    parser = argparse.ArgumentParser(description="Embedding PCA visualisation for ADRIA")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt")
    parser.add_argument("--top-n",      type=int, default=200,
                        help="Number of top-frequency tokens to annotate (default: 200)")
    parser.add_argument("--pca-n",      type=int, default=5000,
                        help="Number of tokens used in full scatter plot (default: 5000)")
    parser.add_argument("--device",     type=str, default="cpu",
                        help="Device (default: cpu — embedding matrix fits easily)")
    parser.add_argument("--out-dir",    type=str, default="figures")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(exist_ok=True)

    print(f"Loading tokenizer ...")
    tokenizer = ByteBPETokenizer.load(TOKENIZER_CONFIG["output"])

    print(f"Loading model from {ckpt_path} ...")
    device = torch.device(args.device)
    model  = load_model(ckpt_path, device)

    # Extract embedding matrix (vocab_size, d_model)
    emb_weight = model.token_emb.token_embeddings.weight.detach().cpu().numpy()
    vocab_size, d_model = emb_weight.shape
    print(f"  Embedding matrix: {vocab_size} x {d_model}")

    # ── PCA on all embeddings ─────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=42)
    coords_all = pca.fit_transform(emb_weight)
    explained = pca.explained_variance_ratio_
    print(f"  PCA variance explained: PC1={explained[0]*100:.1f}%, PC2={explained[1]*100:.1f}%")

    # ── Full scatter (sampled subset for readability) ─────────────────────────
    rng = np.random.default_rng(42)
    sample_idx = rng.choice(vocab_size, size=min(args.pca_n, vocab_size), replace=False)
    coords_sample = coords_all[sample_idx]

    # Colour by token ID as a proxy for "early vocab = byte-level / special" vs "later = BPE merges"
    fig, ax = plt.subplots(figsize=(10, 8))
    sc = ax.scatter(
        coords_sample[:, 0], coords_sample[:, 1],
        c=sample_idx, cmap="plasma", s=2, alpha=0.4, linewidths=0,
    )
    plt.colorbar(sc, ax=ax, label="Token ID")
    ax.set_title(f"Token embedding PCA — ADRIA\n"
                 f"({args.pca_n} of {vocab_size} tokens, "
                 f"PC1={explained[0]*100:.1f}% + PC2={explained[1]*100:.1f}% variance)")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    plt.tight_layout()
    out1 = out_dir / "embedding_pca_full.png"
    plt.savefig(out1, dpi=150)
    plt.close()
    print(f"Saved {out1}")

    # ── Annotated scatter: top-N most-frequent tokens ─────────────────────────
    # "Most frequent" = lowest token ID, since BPE sorts merges by frequency
    top_ids = list(range(2, 2 + args.top_n))   # skip BOS(0) and EOS(1)
    top_coords = coords_all[top_ids]

    fig2, ax2 = plt.subplots(figsize=(14, 10))
    ax2.scatter(top_coords[:, 0], top_coords[:, 1], s=15, c="steelblue", alpha=0.7, zorder=2)

    for tid in top_ids:
        try:
            text = tokenizer.decode([tid])
            display = repr(text)[1:-1][:12]   # truncate long tokens
        except Exception:
            display = f"<{tid}>"
        ax2.annotate(
            display,
            (coords_all[tid, 0], coords_all[tid, 1]),
            fontsize=5, ha="center", va="bottom", alpha=0.8,
        )

    ax2.set_title(f"Top-{args.top_n} tokens — embedding PCA\n"
                  f"PC1={explained[0]*100:.1f}% + PC2={explained[1]*100:.1f}%")
    ax2.set_xlabel("PC 1")
    ax2.set_ylabel("PC 2")
    plt.tight_layout()
    out2 = out_dir / "embedding_pca_annotated.png"
    plt.savefig(out2, dpi=150)
    plt.close()
    print(f"Saved {out2}")

    # ── Cosine similarity of related token groups ─────────────────────────────
    probe_tokens = [" the", " of", " and", " to", " a", " in", "the", "of",
                    " king", " queen", " man", " woman", " good", " bad",
                    "\n", " ", ".", ",", "?", "!"]

    def encode_single(tok_str):
        ids = tokenizer.encode(tok_str, add_bos=False, add_eos=False)
        return ids[0] if ids else None

    probe_ids = [(t, encode_single(t)) for t in probe_tokens]
    probe_ids = [(t, i) for t, i in probe_ids if i is not None]

    if len(probe_ids) >= 4:
        probe_embs = np.stack([emb_weight[i] for _, i in probe_ids])
        # Normalise
        norms = np.linalg.norm(probe_embs, axis=1, keepdims=True)
        probe_embs_n = probe_embs / (norms + 1e-8)
        sim = probe_embs_n @ probe_embs_n.T

        fig3, ax3 = plt.subplots(figsize=(8, 7))
        im = ax3.imshow(sim, vmin=-1, vmax=1, cmap="RdBu_r")
        labels = [t for t, _ in probe_ids]
        ax3.set_xticks(range(len(labels)))
        ax3.set_yticks(range(len(labels)))
        ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax3.set_yticklabels(labels, fontsize=8)
        ax3.set_title("Token embedding cosine similarity — ADRIA")
        plt.colorbar(im, ax=ax3)
        plt.tight_layout()
        out3 = out_dir / "embedding_cosine_sim.png"
        plt.savefig(out3, dpi=150)
        plt.close()
        print(f"Saved {out3}")


if __name__ == "__main__":
    main()
