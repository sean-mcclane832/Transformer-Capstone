from pathlib import Path

MODEL_NAME = "ADRIA"  # Attention-Driven Recursive Inference Architecture — the backronym is in the docs, the real reason isn't.

ROOT = Path(__file__).resolve().parents[1]

# ── Shared across all size tiers ──────────────────────────────────────────
_BASE_CONFIG = {
    "seed":                42,
    "device":              "cuda",
    "vocab_size":          32768,  # tied to the trained tokenizer — never changes between tiers
    "dropout":             0.1,
    "return_attn_weights": False,
    "use_rope":            False,  # set True to enable Rotary Position Embeddings in attention
}

# ── Size presets — architecture-varying keys only ─────────────────────────
# d_ff is always 4 × d_model (GPT-2 convention).
# To add a new tier, append an entry here and update ACTIVE_TIER below.
MODEL_TIERS = {
    "nano": {                   # dev config — overfit sanity test, fast iteration on CPU
        "d_model":     64,
        "n_heads":     4,
        "n_layers":    4,
        "d_ff":        256,
        "max_seq_len": 64,
    },
    "small": {                  # GPT-2 Small scale — first real training run (~117M params)
        "d_model":     768,
        "n_heads":     12,
        "n_layers":    12,
        "d_ff":        3072,
        "max_seq_len": 512,
    },
}

ACTIVE_TIER = "small"           # ← change this one line to switch size tiers

# GENERAL_CONFIG is the single dict imported by every other module.
# Merging base + active tier keeps full backward compatibility — no call sites change.
GENERAL_CONFIG = {**_BASE_CONFIG, **MODEL_TIERS[ACTIVE_TIER]}

# ── Tokenizer ─────────────────────────────────────────────────────────────
TOKENIZER_CONFIG = {
    "input": [
        str(ROOT / "data" / "raw" / "wikitext103.txt"),
        str(ROOT / "data" / "raw" / "pg19.txt"),
        str(ROOT / "data" / "raw" / "openwebtext.txt"),
        str(ROOT / "data" / "raw" / "input.txt"),
        str(ROOT / "data" / "raw" / "greatgatsby.txt"),
    ],
    "output": str(ROOT / "tokenizer" / "tokenizer.json"),
    "min_frequency": 4,
    "max_chars": 5_000_000,  # 5M chars — keeps pure-Python BPE tractable while covering diverse corpus
    "add_special_tokens": True,
}

SCRIPT_CONFIG = {
    "train_tokenizer": {
        "preview_text": "hello world!",
    },
    "test_embedder": {
        "text": "Hello world from the embedder test script.",
        "input_embedding_d_model": 4,
        "input_embedding_vocab_size": 6,
        "positional_d_model": 4,
        "positional_seq_len": 6,
        "positional_sample_seq_len": 3,
        "positional_dropout": 0.0,
        "long_text_multiplier": 8,
    },
    "test_projections": {
        "text": "hello world",
    },
    "test_tokenizer": {
        "samples": [
            "To be, or not to be: that is the question.",
            "Friends, Romans, countrymen, lend me your ears.",
            "hello world",
        ],
    },
    "utf8_demo": {
        "decode_sample_ids": [126],
    },
}
