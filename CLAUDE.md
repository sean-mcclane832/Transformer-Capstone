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
  → Embedding dropout
  → N × Transformer Block (Pre-LN):
      - LayerNorm → Masked Multi-Head Self-Attention → Residual
      - LayerNorm → Position-wise Feed-Forward Network → Residual
  → Final LayerNorm
  → LM head: linear projection (d_model → vocab_size)
  → (loss: cross-entropy on shifted targets)
  → Softmax → probability distribution (inference-only)
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

Existing directories are listed first; planned directories (not yet created) are marked with `🔲`.

```
Transformer-Capstone/
├── attention/
│   ├── projections.py          # AttentionProjections (W_q, W_k, W_v)
│   ├── scaled_dot.py           # scaled_dot_product_attention (plain function)
│   ├── softmax.py              # numerically-stable softmax (custom)
│   ├── mask.py                 # causal_mask() plain function
│   └── multi_head.py           # MultiHeadAttention module
├── transformer/                🔲 # planned — block-level modules
│   ├── feed_forward.py         🔲 # FeedForward (FFN) module
│   ├── layer_norm.py           🔲 # custom LayerNorm
│   └── block.py                🔲 # TransformerBlock (Pre-LN)
├── model/                      🔲 # planned — full-model assembly
│   ├── gpt.py                  🔲 # GPT class (embeddings + blocks + LM head)
│   └── generate.py             🔲 # sampling functions (greedy, temp, top-k, top-p)
├── data/
│   ├── raw/
│   │   ├── input.txt           # Shakespeare corpus
│   │   └── greatgatsby.txt     # Project Gutenberg novel
│   ├── prepare.py              🔲 # tokenize corpus → train.pt / val.pt
│   ├── train.pt                🔲 # pre-tokenized training tokens
│   ├── val.pt                  🔲 # pre-tokenized validation tokens
│   └── dataset.py              🔲 # TokenDataset (sliding window)
├── scripts/
│   ├── train_tokenizer.py      # CLI: trains and saves BPE tokenizer
│   ├── test_embedder.py        # Unit tests for embedding pipeline
│   ├── test_projections.py     # Unit tests for Q/K/V projections
│   ├── test_tokenizer.py       # Unit tests for tokenizer
│   ├── test_multi_head.py      # Unit tests for MultiHeadAttention
│   ├── test_feed_forward.py    🔲
│   ├── test_layer_norm.py      🔲
│   ├── test_block.py           🔲
│   ├── test_model.py           🔲
│   ├── test_dataset.py         🔲
│   └── plot_curves.py          🔲 # training curve visualizations
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
├── checkpoints/                🔲 # saved model checkpoints during training
├── figures/                    🔲 # output PNGs of training curves, attention maps
├── train.py                    🔲 # training loop entrypoint
├── cli.py                      🔲 # CLI: prompt → generated text
├── CLAUDE.md                   # This file
└── README.md
```

---

## Current Configuration (`utils/config.py`)

Active dev settings:

```python
GENERAL_CONFIG = {
    "seed": 42,
    "device": "cpu",
    "vocab_size": 8192,
    "d_model": 64,
    "n_heads": 4,
    "max_seq_len": 64,
    "dropout": 0.0,
    "return_attn_weights": True,
}
```

Values to add as new components are built:

```python
"n_layers": 4,        # number of stacked transformer blocks
"d_ff": 256,          # FFN inner dimension; convention is 4 * d_model
"batch_size": 64,     # used by DataLoader during training
```

These small values are intentional for fast iteration during development.
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
- Note: full-dimensional projections. Head splitting happens downstream in `MultiHeadAttention`.

### `attention/softmax.py` — `softmax`
- Plain function with subtract-max numerical stability trick before `exp`
- No nn.Module; consumed by `scaled_dot_product_attention`

### `attention/scaled_dot.py` — `scaled_dot_product_attention`
- Plain function, no learnable parameters — dropout is passed in from `MultiHeadAttention` as an `nn.Dropout` instance
- Scales by `sqrt(d_k)` where `d_k = Q.size(-1)` — inferred from input, no hardcoded config value
- Accepts optional `mask` and `dropout` arguments; mask applied via `masked_fill(mask == 0, -inf)` before softmax
- Signature: `scaled_dot_product_attention(Q, K, V, mask=None, dropout=None)` → `(output, weights)`
- Expected input shape: `(batch, n_heads, seq_len, d_k)`

### `attention/mask.py` — `causal_mask`
- Plain function: `causal_mask(seq_len)` → lower-triangular `(seq_len, seq_len)` tensor of ones
- No learnable parameters; shape must be broadcast-compatible with `(batch, n_heads, seq_len, seq_len)`

### `attention/multi_head.py` — `MultiHeadAttention`
- `nn.Module` wrapping `AttentionProjections`, head splitting/recombination, `scaled_dot_product_attention`, and `W_o`
- Reshape pattern: `(batch, seq_len, d_model)` → `view(...)` → `transpose(1, 2)` → `(batch, n_heads, seq_len, d_k)`
- Auto-builds causal mask when `mask` argument is `None`
- `return_attn_weights` flag (from config) controls whether forward returns `output` or `(output, weights)`
- Owns the `nn.Dropout` instance for attention weights; passed into scaled-dot only during `self.training`
- Verified with `scripts/test_multi_head.py`: output shape, weights shape, weights sum-to-one per row, causal mask zeroes upper triangle

### `utils/seed.py` — `set_seed`
- `set_seed(seed)` — sets `random`, `numpy`, `torch`, and `torch.cuda` seeds plus `PYTHONHASHSEED`; enables `cudnn.deterministic` and disables `cudnn.benchmark`

---

## What Has NOT Been Built Yet

Implement these in order. Each step has a corresponding test script in `scripts/`.

1. **FeedForward (`transformer/feed_forward.py`)** — two linear layers with GELU activation, inner dim `d_ff = 4 * d_model`, dropout after activation
2. **Custom LayerNorm (`transformer/layer_norm.py`)** — hand-implemented `(x - mean) / sqrt(var + eps) * γ + β` over `dim=-1`; verified against `nn.LayerNorm`
3. **TransformerBlock (`transformer/block.py`)** — Pre-LN: `x = x + dropout(attn(LN(x)))` then `x = x + dropout(ffn(LN(x)))`
4. **GPT model (`model/gpt.py`)** — embeddings + N blocks + final LayerNorm + LM head; weight tying between input embedding and LM head; GPT-2-style weight init (`N(0, 0.02)`, residual projections scaled by `1/sqrt(2 * n_layers)`)
5. **Data pipeline (`data/prepare.py`, `data/dataset.py`)** — pre-tokenize corpus to flat tensor, save to `train.pt` / `val.pt`; `TokenDataset` produces `(x, y)` pairs shifted by one position; 90/10 train/val split
6. **Training loop (`train.py`)** — AdamW with parameter-group weight decay, linear warmup + cosine LR schedule, gradient clipping at `max_norm=1.0`, cross-entropy loss with reshape, validation every N steps, checkpointing, optional fp16 mixed precision
7. **Sanity overfit test** — train on 5 fixed batches with `dropout=0.0`; loss must drop to <0.5 within ~1000 steps. Mandatory before any full run.
8. **Generation (`model/generate.py`)** — greedy, temperature, top-k, top-p (nucleus); combined sampler under `torch.no_grad()` and `model.eval()`
9. **CLI (`cli.py`)** — argparse: `--checkpoint`, `--prompt`, `--max-new-tokens`, `--temperature`, `--top-k`, `--top-p`
10. **Visualization (`scripts/plot_curves.py`)** — loss/perplexity/LR curves; optional attention heatmaps

---

## Critical Implementation Notes

### Scaled dot-product attention
- Plain function, no learnable parameters — dropout passed in as an `nn.Dropout` instance
- Scales by `sqrt(d_k)` inferred from `Q.size(-1)`
- Causal mask applied via `masked_fill(mask == 0, -inf)` before softmax
- Mask shape must broadcast over `(batch, n_heads, seq_len, seq_len)`

### Multi-head attention
- Split via reshape + transpose: `(batch, seq_len, d_model)` → `(batch, n_heads, seq_len, d_k)`
- Concatenate heads: transpose back → `.contiguous().view(batch, seq_len, d_model)` (the `.contiguous()` is required after transpose)
- Apply `W_o: (d_model, d_model)`, `bias=False`

### FeedForward
- Activation: **GELU** (matches GPT-2). Not ReLU.
- Inner dimension: `d_ff = 4 * d_model` (convention)
- `bias=True` on both linear layers (different from attention)
- Dropout placement: after activation, before `W_2`
- No nonlinearity after `W_2` — block ends linear

### LayerNorm
- Normalize over **feature dimension only** (`dim=-1`), per token, independent of batch/seq
- Use `unbiased=False` for variance (matches `nn.LayerNorm` default)
- `eps=1e-5`
- `γ` initialized to ones, `β` to zeros, both shape `(d_model,)`

### Pre-LN transformer block
- Pattern: `x = x + sublayer(LayerNorm(x))` for both attention and FFN
- The residual stream is never normalized in place — gradients flow through residuals without passing through any LN
- `ln1` and `ln2` are separate modules, never shared
- Residual dropout applied to sublayer output before the add

### GPT model assembly
- Final LayerNorm before the LM head is mandatory (residual stream scale drifts with depth otherwise)
- LM head outputs raw logits — softmax happens inside the loss function
- **Weight tying:** `self.lm_head.weight = self.token_emb.token_embeddings.weight` — halves param count contribution and slightly improves perplexity
- **Weight init:** all `nn.Linear` and `nn.Embedding` weights `N(0, 0.02)`, biases zero. Residual projections (`W_o` in attention, `W_2` in FFN) scaled by `1/sqrt(2 * n_layers)` to keep residual stream variance stable with depth

### Data pipeline
- Pre-tokenize once, save as `torch.long` tensor; do not tokenize on-the-fly during training
- Bulk corpus tokenized **without** BOS/EOS — the sliding window doesn't respect document boundaries
- `TokenDataset.__len__` = `len(ids) - seq_len - 1` (off-by-one matters: need both `x[i:i+seq_len]` and `y[i+1:i+seq_len+1]`)
- Targets are inputs shifted by one position; loss is computed at every position in parallel
- 90/10 split, validation block contiguous (do not shuffle the split itself)

### Training loop
- **Optimizer:** AdamW (not Adam). `betas=(0.9, 0.95)`, `weight_decay=0.1`, `eps=1e-8`
- **Parameter groups:** weight decay only on tensors with `dim() >= 2`. Skip decay on LayerNorm weights, biases, embeddings.
- **LR schedule:** linear warmup (`~100-1000` steps) → cosine decay to `min_lr = max_lr / 10`. Apply manually each step.
- **Grad clip:** `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `optimizer.step()`
- **Loss:** `F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))` — reshape required, expects raw logits
- **Perplexity:** `exp(loss)` for logging
- **Checkpoint cadence:** every N steps + best-val-loss; keep last 3 + best
- **Mixed precision:** `torch.cuda.amp.autocast(dtype=torch.float16)` + `GradScaler` once on GPU; unscale before grad clip

### Sanity overfit test
- Run before any full training run
- 5 fixed batches looped for ~2000 steps with `dropout=0.0`
- Expected: loss drops from `~log(vocab_size) ≈ 9.0` to `<0.5`
- If it doesn't:
  - Stuck near 9.0 → forward broken or backward not connecting
  - Drops then NaN → exploding grads (check clip, init)
  - Drops slowly → LR/optimizer/init issue

### Generation
- Always under `torch.no_grad()` and `model.eval()`
- Crop input to last `max_seq_len` tokens before each forward pass (positional encoding will raise otherwise)
- Only the last position's logits are used per step: `logits[:, -1, :]`
- **Temperature** before any filtering: `logits / T`
- **Top-k:** keep top `k`, set rest to `-inf`. Typical `k=40-50`.
- **Top-p:** sort, take cumulative probs, keep nucleus where `cumprob ≤ p`. Typical `p=0.9`.
- Combine: temperature → top-k → top-p → softmax → `multinomial`
- Stop on EOS or max_new_tokens

---

## Target Scale

Development config: `d_model=64`, 2–4 heads, 2–4 layers, `seq_len=64`

Practical training target:
```
d_model     : 384–768
n_heads     : 6–12
n_layers    : 6–12
d_ff        : 4 * d_model
seq_len     : 256–512
batch_size  : 32–64
dropout     : 0.1
precision   : fp16 (mixed precision via torch.cuda.amp)
```

Aspirational ceiling (RTX 5070):
```
GPT-2 Small  : d_model=768, 12 heads, 12 layers, seq_len=1024, ~117M params
GPT-2 Medium : d_model=1024, 16 heads, 24 layers, ~345M params
```

---

## Deferred Features

**TurboQuant KV cache quantization** — post-training inference optimization. Implemented as `TurboQuantKVCache` inside the attention layer (Option B). Use the `prod` variant from the paper (unbiased for inner-product estimation, which is exactly what `Q · K^T` is). Do not implement until the full model is trained and generating coherent text.

**Web UI** — CLI is sufficient for the capstone. Web frontend (Flask/FastAPI server or ONNX-Runtime-Web) is bonus only.

**Attention visualization** — heatmap of `(seq, seq)` attention weights per head/layer. Cool for the writeup; not on the critical path.

---

## Milestones

| # | Goal | Status |
|---|---|---|
| 1 | BPE tokenizer with BOS/EOS | ✅ Complete |
| 2 | Token embeddings + positional encoding | ✅ Complete |
| 3 | Q/K/V projection layers | ✅ Complete |
| 4 | Custom softmax | ✅ Complete |
| 5 | Scaled dot-product attention + causal mask | ✅ Complete |
| 6 | Full MultiHeadAttention module | ✅ Complete |
| 7 | FeedForward (FFN) module | 🔲 Next |
| 8 | Custom LayerNorm | 🔲 |
| 9 | Full transformer block (Pre-LN, unit-tested) | 🔲 |
| 10 | GPT model assembly + weight init + tying | 🔲 |
| 11 | Data pipeline (pre-tokenize, dataset, dataloader) | 🔲 |
| 12 | Training loop (AdamW + warmup/cosine + grad clip) | 🔲 |
| 13 | Sanity overfit test on tiny dataset | 🔲 |
| 14 | Full training run with loss/perplexity logging | 🔲 |
| 15 | Generation (greedy + temperature + top-k + top-p) | 🔲 |
| 16 | CLI interface | 🔲 |
| 17 | Training curve visualizations | 🔲 |
| 18 | Capstone documentation (math + architecture writeup) | 🔲 |
| 19 | TurboQuant KV cache (deferred bonus) | 🔲 |

---

## Capstone Documentation Structure

For Milestone 18, the writeup should include:

1. **Architectural derivation** — every component mathematically. Embeddings, the QK^T/√d_k story, masking, residuals, LayerNorm, FFN. Cite Vaswani et al. and Radford et al.
2. **Implementation choices** — Pre-LN vs Post-LN, AdamW vs Adam, weight tying, init scheme, BPE vs char. Why each choice.
3. **Training methodology** — dataset, tokenization, splits, hyperparameters, hardware, runtime
4. **Results** — loss curves, perplexity, sample generations at different temperatures/sampling strategies
5. **Failure modes & limitations** — repetition, drift, factual errors, context length. Honest assessment.
6. **Reflection** — what scaling would buy, what you'd change, why "just use HuggingFace" misses the point

---

## Reference Material

- **Andrej Karpathy** — *Let's build GPT: from scratch, in code, spelled out* — https://youtube.com/watch?v=kCc8FmEb1nY (primary implementation reference)
- **Umar Jamil** — *Coding a Transformer from scratch on PyTorch* — https://youtube.com/watch?v=ISNdQcPhsts (strong on embeddings/positional encoding)
- **Vaswani et al. (2017)** — *Attention Is All You Need* — original architecture paper
- **Radford et al. (2019)** — *Language Models are Unsupervised Multitask Learners* — GPT-2 paper, source of Pre-LN, init scheme, weight tying conventions
- **nanoGPT** (Karpathy) — https://github.com/karpathy/nanoGPT — reference for parameter-group weight decay, mixed precision, training loop structure

---

## Notes for Claude

- **Project knowledge does not auto-sync.** If searches return stale results after new code is merged, manually re-sync the GitHub connection in project settings.
- Always check `utils/config.py` for current hyperparameter values before generating code that uses `d_model`, `vocab_size`, `max_seq_len`, `n_heads`, `n_layers`, `d_ff`, or `dropout`.
- All new modules should import config values from `utils/config.py` rather than hardcoding.
- Follow existing file naming and module structure conventions. New blocks live in `transformer/`, full model in `model/`, data in `data/`.
- **Module patterns:**
  - Plain function: `softmax`, `scaled_dot_product_attention`, `causal_mask`
  - `nn.Module` class: `MultiHeadAttention`, `FeedForward`, `LayerNorm`, `TransformerBlock`, `GPT`
- **Pre-LN** is the convention for this project. Never generate Post-LN code.
- **GELU** is the FFN activation. Never ReLU.
- **AdamW** is the optimizer. Never Adam.
- For each new module, also generate a corresponding `scripts/test_<name>.py` that follows the existing pattern in `test_multi_head.py`: imports, `make_module()` helper, `make_input()` helper, named test cases that raise `AssertionError` on failure, pass/fail summary at end.
- The project is intentionally educational. Prefer clear, well-commented implementations over clever one-liners.
- When in doubt about a design choice, default to whatever GPT-2 does. nanoGPT is the cleanest reference implementation.
- Sean prefers to be addressed as "Handsome" per project preference.
