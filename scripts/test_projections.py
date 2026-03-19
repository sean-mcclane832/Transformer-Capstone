import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attention.projections import AttentionProjections
from text_processing.text_processor import TextEmbedder
from utils.config import GENERAL_CONFIG, SCRIPT_CONFIG, TOKENIZER_CONFIG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test query, key, and value projections.")
    parser.add_argument(
        "--tokenizer",
        default=TOKENIZER_CONFIG["output"],
        help="Path to tokenizer JSON file.",
    )
    parser.add_argument(
        "--text",
        default=SCRIPT_CONFIG["test_projections"]["text"],
        help="Text to embed before projection.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=GENERAL_CONFIG["d_model"],
        help="Shared model width for the embedder and projections.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=GENERAL_CONFIG["vocab_size"],
        help="Tokenizer vocabulary size.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=GENERAL_CONFIG["max_seq_len"],
        help="Maximum sequence length for positional encoding.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=GENERAL_CONFIG["dropout"],
        help="Dropout used by the text embedder positional encoding.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    encoder = TextEmbedder(
        tokenizer_path=args.tokenizer,
        d_model=args.d_model,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        dropout=args.dropout,
    )
    proj = AttentionProjections(d_model=args.d_model)

    x = encoder.embed_text(args.text)
    q, k, v = proj(x)

    print(f"input shape: {tuple(x.shape)}")
    print(f"Q shape: {tuple(q.shape)}")
    print(f"K shape: {tuple(k.shape)}")
    print(f"V shape: {tuple(v.shape)}")


if __name__ == "__main__":
    main()
