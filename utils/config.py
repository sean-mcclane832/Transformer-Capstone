from pathlib import Path

MODEL_NAME = "ADRIA"  # Attention-Driven Recursive Inference Architecture — the backronym is in the docs, the real reason isn't.

ROOT = Path(__file__).resolve().parents[1]

GENERAL_CONFIG = {
    "seed": 42,
    "device": "cuda",
    "vocab_size": 32768,
    "d_model": 768,
    "n_heads": 12,
    "n_layers": 12,
    "max_seq_len": 512,
    "dropout": 0.1,
    "return_attn_weights": False,
    "d_ff": 3072,           # 4 × d_model
}

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
