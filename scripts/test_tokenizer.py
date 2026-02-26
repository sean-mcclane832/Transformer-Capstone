import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tokenizer.token_class import ByteBPETokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a trained tokenizer on sample text.")
    parser.add_argument(
        "--tokenizer",
        default=str(ROOT / "tokenizer" / "tokenizer.json"),
        help="Path to tokenizer JSON file.",
    )
    parser.add_argument(
        "--text",
        default=None,
        help="Optional custom text to tokenize. If omitted, built-in examples are used.",
    )
    args = parser.parse_args()

    tok = ByteBPETokenizer.load(args.tokenizer)

    samples = [args.text] if args.text is not None else [
        "To be, or not to be: that is the question.",
        "Friends, Romans, countrymen, lend me your ears.",
        "Hello World 😄",
    ]

    print(f"Loaded tokenizer: {args.tokenizer}")
    print(f"vocab_size={tok.vocab_size}, bos_id={tok.bos_id}, eos_id={tok.eos_id}")
    print()

    for i, text in enumerate(samples, start=1):
        ids = tok.encode(text, add_bos=(tok.bos_id is not None), add_eos=(tok.eos_id is not None))
        decoded = tok.decode(ids)

        print(f"Sample {i}:")
        print(f"  text    : {ascii(text)}")
        print(f"  ids     : {ids}")
        print(f"  n_tokens: {len(ids)}")
        print(f"  decoded : {ascii(decoded)}")
        print(f"  match   : {decoded == text}")
        print()


if __name__ == "__main__":
    main()
