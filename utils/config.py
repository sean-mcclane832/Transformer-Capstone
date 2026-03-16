from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

GENERAL_CONFIG = {
    'vocab_size': 8192
}

TOKENIZER_CONFIG = {
    "input": [
        str(ROOT / "data" / "raw" / "input.txt"),
        str(ROOT / "data" / "raw" / "greatgatsby.txt"),
    ],
    "output": str(ROOT / "tokenizer" / "tokenizer.json"),
    "min_frequency": 4,
    "max_chars": 0,
    "add_special_tokens": True,
}
