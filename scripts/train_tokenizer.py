import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizer.token_class import ByteBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a byte-level BPE tokenizer.")
    parser.add_argument(
        "--input",
        default=str(ROOT / "data" / "raw" / "input.txt"),
        help="Path to training text file.",
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "tokenizer" / "tokenizer.json"),
        help="Path to save tokenizer JSON.",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=4096,
        help="Target tokenizer vocabulary size.",
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

    tok = ByteBPETokenizer(
        vocab_size=args.vocab_size,
        add_special_tokens=not args.no_special_tokens,
    )
    tok.train(text)
    tok.save(str(output_path))

    tok2 = ByteBPETokenizer.load(str(output_path))
    ids = tok2.encode(
        "hello 😄",
        add_bos=not args.no_special_tokens,
        add_eos=not args.no_special_tokens,
    )
    print(ids[:20])
    print(ascii(tok2.decode(ids)))


if __name__ == "__main__":
    main()

