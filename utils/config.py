from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

GENERAL_CONFIG = {
    "seed": 42,
    "device": "cpu",
    "vocab_size": 8192,
    "d_model": 64,
    "n_heads": 4,
    "n_layers": 4,
    "max_seq_len": 64,
    "dropout": 0.1,
    "return_attn_weights": False,
    "d_ff": 256
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
    "max_chars": 100_000_000,  # 100M chars — representative sample, keeps pure-Python BPE tractable
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
