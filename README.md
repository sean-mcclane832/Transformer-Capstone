## Transformer Capstone (Pure NumPy GPT-2 style)

This project includes a **fully trainable GPT-2 style language model written without PyTorch or TensorFlow**.

## What is implemented

- UTF-8 byte-level BPE tokenizer (default vocab: **32,000**)
- Decoder-only Transformer with GPT-2 block structure
- GPT-2-sized presets:
  - `gpt2` (12L, 12H, 768D)
  - `gpt2-medium` (24L, 16H, 1024D)
  - `gpt2-large` (36L, 20H, 1280D)
  - `gpt2-xl` (48L, 25H, 1600D)
- Pure NumPy forward/backward for:
  - embeddings
  - layer norm
  - causal self-attention
  - MLP + GELU
  - tied LM head
- Pure NumPy AdamW optimizer
- Training + generation scripts

## Install

```bash
pip install -r requirements.txt
```

## Train

```bash
python scripts/train.py \
  --train-text data/train.txt \
  --out-dir artifacts/gpt2_32k \
  --model-size gpt2 \
  --vocab-size 32000 \
  --block-size 1024 \
  --batch-size 2 \
  --steps 200
```

Outputs:

- `artifacts/gpt2_32k/tokenizer.json`
- `artifacts/gpt2_32k/weights.npz`
- `artifacts/gpt2_32k/meta.json`

## Generate

```bash
python scripts/generate.py \
  --checkpoint-dir artifacts/gpt2_32k \
  --prompt "Once upon a time" \
  --max-new-tokens 100 \
  --temperature 0.9 \
  --top-k 50
```

## Notes

- GPT-2 presets match the architectural depth/head/width settings.
- With vocab fixed to 32k, total parameter counts are close in scale to GPT-2 but not identical to OpenAI's original vocab sizing.
- Training the full `gpt2` preset in pure NumPy is compute-heavy; use `gpt2-tiny` for quick local smoke tests.
