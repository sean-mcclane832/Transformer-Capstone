# CLAUDE.md — Transformer Capstone (MiniGPT)

This file is the authoritative context document for AI-assisted development on this project.
Read it fully before generating any code, suggestions, or analysis.

---

## Project Identity

**Repository:** Transformer-Capstone
**Project type:** Senior capstone — decoder-only GPT-style transformer built from scratch in PyTorch
**Goal:** Implement the full language model pipeline from tokenizer through autoregressive generation, understanding every component at the architectural and mathematical level
**Guiding principle:** No black-box abstractions. Every module is hand-implemented. The value is comprehension, not convenience.

---

## Architecture Decision

This is a **decoder-only transformer** (GPT-2 / GPT-3 style). There is no encoder and no cross-attention. The model performs **next-token prediction** via self-supervised training. Loss function is cross-entropy over the vocabulary.

Full architecture stack:

```
Raw text
  → BPE tokenizer (trained on dataset)
  → Token IDs + BOS/EOS tokens
  → Token embedding matrix (learned)
  → Sinusoidal positional encoding (fixed)
  → N × Transformer Block:
      - Masked Multi-Head Self-Attention
      - Residual connection + Layer Normalization
      - Position-wise Feed-Forward Network
      - Residual connection + Layer Normalization
  → Final linear projection (d_model → vocab_size)
  → Softmax → probability distribution
  → Autoregressive generation loop
```

---

## Hardware

| Device | VRAM | Role |
|---|---|---|
| RTX 2060 Super | 8 GB | Development and unit testing |
| RTX 5070 | 12 GB | Full training runs |

---

## Repository Structure

```
Transformer-Capstone/
├── attention/
│   ├── projections.py          # AttentionProjections (W_q, W_k, W_v)
│   ├── scaled_dot.py           # scaled_dot_product_attention (plain function)
│   └── mask.py                 # causal_mask() plain function
├── data/
│   └── raw/
│       ├── input.txt           # Shakespeare corpus
│       └── greatgatsby.txt     # Project Gutenberg novel
├── scripts/
│   ├── train_tokenizer.py      # CLI: trains and saves BPE tokenizer
│   ├── test_embedder.py        # Unit tests for embedding pipeline
│   ├── test_projections.py     # Unit tests for Q/K/V projections
│   └── test_tokenizer.py       # Unit tests for tokenizer
├── text_processing/
│   ├── token_class.py          # ByteBPETokenizer class
│   ├── embedding_classes.py    # InputEmbeddings, PositionalEncoding
│   ├── text_processor.py       # TextEmbedder pipeline (tokenize → embed)
│   └── utf-8.py                # Early scratch/reference BPE demo (not used in pipeline)
├── tokenizer/
│   └── tokenizer.json          # Saved trained BPE tokenizer
├── utils/
│   ├── config.py               # GENERAL_CONFIG, TOKENIZER_CONFIG, SCRIPT_CONFIG
│   ├── seed.py                 # set_seed() — deterministic seeding helper
│   ├── helpers.py              # Placeholder (empty)
│   └── io.py                   # Placeholder (empty)
├── CLAUDE.md                   # This file
└── README.md
```

---

## Current Configuration (`utils/config.py`)

These are the **active dev settings** used across all modules:

```python
GENERAL_CONFIG = {
    "seed": 42,
    "device": "cpu",
    "vocab_size": 8192,
    "d_model": 64,
    "max_seq_len": 64,
    "dropout": 0.0,
}
```

These values are small on purpose — fast iteration during development.
Training runs will scale up significantly (see Target Scale below).

---

## Completed Components

### `text_processing/token_class.py` — `ByteBPETokenizer`
- Byte-level BPE tokenizer trained on Shakespeare + Great Gatsby
- `vocab_size = 8192`
- Supports `<bos>` and `<eos>` special tokens
- `.encode(text, add_bos, add_eos)` → list of token IDs
- `.decode(ids)` → string
- Serializable via `.save()` / `.load()`

### `text_processing/embedding_classes.py`
- `InputEmbeddings(d_model, vocab_size)` — learned `nn.Embedding` lookup, scaled by `sqrt(d_model)` per the original paper
- `PositionalEncoding(d_model, seq_len, dropout)` — sinusoidal encoding stored as a non-learnable buffer via `register_buffer`; raises `ValueError` if input exceeds `seq_len`

### `text_processing/text_processor.py` — `TextEmbedder`
- Combines tokenizer + embeddings + positional encoding into a single pipeline
- `embed_text(text)` → tensor of shape `(1, seq_len, d_model)`

### `attention/projections.py` — `AttentionProjections`
- Three `nn.Linear` layers: `W_q`, `W_k`, `W_v`, each `(d_model, d_model)`
- `forward(x)` → `(Q, K, V)` each of shape `(batch, seq_len, d_model)`
- Note: these are **full-dimensional** projections. Head splitting happens downstream in `MultiHeadAttention`.

### `attention/scaled_dot.py` — `scaled_dot_product_attention`
- Plain function, no learnable parameters — dropout is passed in from `MultiHeadAttention` as an `nn.Dropout` instance
- Scales by `sqrt(d_k)` where `d_k = Q.size(-1)` — inferred from input, no hardcoded config value
- Accepts optional `mask` and `dropout` arguments; mask applied via `masked_fill(mask == 0, -inf)` before softmax
- Signature: `scaled_dot_product_attention(Q, K, V, mask=None, dropout=None)` → `(output, weights)`
- Expected input shape: `(batch, n_heads, seq_len, d_k)`

### `attention/mask.py` — `causal_mask`
- Plain function: `causal_mask(seq_len)` → lower-triangular `(seq_len, seq_len)` tensor of ones
- No learnable parameters; shape must be broadcast-compatible with `(batch, n_heads, seq_len, seq_len)` when used in multi-head attention

### `utils/seed.py` — `set_seed`
- `set_seed(seed)` — sets `random`, `numpy`, `torch`, and `torch.cuda` seeds plus `PYTHONHASHSEED`; enables `cudnn.deterministic` and disables `cudnn.benchmark`

---

## What Has NOT Been Built Yet

Implement these in order:

1. **Multi-head splitting and parallel attention** — reshape Q/K/V for `n_heads`; wire `ScaledDotAttention` + `causal_mask` together
2. **Head concatenation + output projection (W_O)** → full `MultiHeadAttention` module
3. **Feed-forward block** — two linear layers with ReLU/GELU, inner dim = `4 * d_model`
4. **Full transformer block** — attention + FFN + residual connections + layer norm
5. **Model assembly** — embedder + N blocks + final projection
6. **Training loop** — DataLoader, forward pass, cross-entropy loss, backward, optimizer, LR scheduling, gradient clipping, logging
7. **Generation function** — greedy / temperature / top-k / nucleus sampling
8. **CLI or minimal interface** — prompt in, generated text out

---

## Critical Implementation Notes

### Scaled dot-product attention
- Plain function `scaled_dot_product_attention` in `attention/scaled_dot.py` — no learnable parameters
- Scales by `sqrt(d_k)` inferred from `Q.size(-1)`; works correctly once Q/K/V are split per head
- `dropout` is an `nn.Dropout` instance owned by `MultiHeadAttention` and passed in at call time
- Causal mask is generated by `causal_mask()` in `attention/mask.py`; shape must broadcast over `(batch, n_heads, seq_len, seq_len)`
- Mask convention: `mask == 0` positions are filled with `-inf` before softmax

### Multi-head attention
- Split Q, K, V by reshaping: `(batch, seq_len, d_model)` → `(batch, n_heads, seq_len, d_k)`
- Run scaled dot-product attention in parallel across all heads
- Concatenate heads: `(batch, n_heads, seq_len, d_k)` → `(batch, seq_len, d_model)`
- Apply output projection `W_O: (d_model, d_model)`

### Layer normalization placement
- Use **Pre-LN** (normalize before attention/FFN, not after) — more stable during training than the original Post-LN paper formulation

### Residual connections
- Every sub-layer (attention, FFN) must be wrapped: `x = x + sublayer(norm(x))`

### Gradient stability
- Clip gradients (`max_norm=1.0`) before the optimizer step
- Use a warmup LR schedule; a cosine decay after warmup is standard

---

## Target Scale

Development config: `d_model=64`, 2–4 heads, 2–4 layers, `seq_len=64`

Practical training target:
```
d_model   : 512–768
n_heads   : 8–12
n_layers  : 6–12
seq_len   : 512
precision : fp16 (mixed precision via torch.cuda.amp)
```

Aspirational ceiling (RTX 5070):
```
GPT-2 Small : d_model=768, 12 heads, 12 layers, seq_len=1024, ~117M params
GPT-2 Medium: d_model=1024, 16 heads, 24 layers, ~345M params
```

---

## Deferred Features

**TurboQuant KV cache quantization** — post-training inference optimization. Implemented as `TurboQuantKVCache` inside the attention layer (Option B). Do not implement until the full model is trained and generating coherent text.

---

## Milestones

| # | Goal | Status |
|---|---|---|
| 1 | BPE tokenizer with BOS/EOS | ✅ Complete |
| 2 | Token embeddings + positional encoding | ✅ Complete |
| 3 | Q/K/V projection layers | ✅ Complete |
| 4 | Scaled dot-product attention + causal mask | ✅ Complete |
| 5 | Full MultiHeadAttention module | 🔲 Next |
| 6 | Feed-forward block | 🔲 |
| 7 | Full transformer block (single, unit-tested) | 🔲 |
| 8 | Stacked blocks + valid full forward pass | 🔲 |
| 9 | Overfit on tiny dataset (sanity check) | 🔲 |
| 10 | Full training run with loss/perplexity logging | 🔲 |
| 11 | Generation function (greedy + sampling) | 🔲 |
| 12 | CLI / minimal interface | 🔲 |
| 13 | Training curve visualizations | 🔲 |
| 14 | Capstone documentation (math + architecture writeup) | 🔲 |

---

## Reference Material

- **Andrej Karpathy** — "Let's build GPT: from scratch, in code, spelled out"
  `https://youtube.com/watch?v=kCc8FmEb1nY` — primary implementation reference
- **Umar Jamil** — "Coding a Transformer from scratch on PyTorch"
  `https://youtube.com/watch?v=ISNdQcPhsts` — strong on embeddings and positional encoding
- **Vaswani et al. (2017)** — "Attention Is All You Need" — original architecture paper

---

## Notes for Claude

- **Project knowledge does not auto-sync.** If searches return stale results after new code is merged, manually re-sync the GitHub connection in project settings.
- Always check `utils/config.py` for current hyperparameter values before generating code that uses `d_model`, `vocab_size`, `max_seq_len`, or `dropout`.
- All new modules should import config values from `utils/config.py` rather than hardcoding.
- Follow existing file naming and module structure conventions.
- When generating attention code, default to the **plain function** pattern for `scaled_dot_product_attention` and the **`nn.Module` class** pattern for `MultiHeadAttention`.
- The project is intentionally educational. Prefer clear, well-commented implementations over clever one-liners.
