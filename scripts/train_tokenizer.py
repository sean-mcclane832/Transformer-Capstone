import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from text_processing.token_class import ByteBPETokenizer
from utils.config import TOKENIZER_CONFIG, GENERAL_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument(
        "--input",
        default=TOKENIZER_CONFIG["input"],
        help="Path to training text file.",
    )
    parser.add_argument(
        "--output",
        default=TOKENIZER_CONFIG["output"],
        help="Path to save tokenizer JSON.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=GENERAL_CONFIG["vocab_size"],
        help="Target tokenizer vocabulary size.",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=TOKENIZER_CONFIG["min_frequency"],
        help="Stop adding merges when best pair frequency drops below this threshold.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=TOKENIZER_CONFIG["max_chars"],
        help="Use only the first N characters of input for faster training (0 = use full file).",
    )
    parser.add_argument(
        "--no-special-tokens",
        action="store_true",
        help="Disable <bos>/<eos> special tokens.",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    text = input_path.read_text(encoding="utf-8")
    if args.max_chars > 0:
        text = text[: args.max_chars]
    print(
        f"Training on {len(text):,} chars | vocab_size={args.vocab_size} | "
        f"min_frequency={args.min_frequency} | special_tokens={not args.no_special_tokens}"
    )

    add_special_tokens = TOKENIZER_CONFIG["add_special_tokens"] and not args.no_special_tokens
    tok = ByteBPETokenizer(
        vocab_size=args.vocab_size,
        add_special_tokens=add_special_tokens,
    )
    tok.train(text, min_frequency=args.min_frequency)
    tok.save(str(output_path))
    print(f"Saved tokenizer to: {output_path}")

    tok2 = ByteBPETokenizer.load(str(output_path))
    ids = tok2.encode(
        "hello world!",
        add_bos=not args.no_special_tokens,
        add_eos=not args.no_special_tokens,
    )
    print(ids[:20])
    print(ascii(tok2.decode(ids)))


if __name__ == "__main__":
    main()
