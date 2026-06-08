"""
expand_corpus.py — append new text files to existing processed corpus

Tokenizes only the specified new files, then concatenates with the
existing train.pt / val.pt. Does NOT re-tokenize anything already
in the corpus.

The pure-Python BPE tokenizer is superlinear in chunk size, so files
are encoded in CHUNK_SIZE slices (same as prepare.py) for realistic
throughput (~6 KB/s vs ~2 KB/s for whole-file encoding).

Usage:
    python data/expand_corpus.py data/raw/newbook1.txt data/raw/newbook2.txt

Project Gutenberg files: pass --strip-gutenberg to remove the standard
header/footer boilerplate before tokenizing.
"""

import sys
import re
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm
from text_processing.token_class import ByteBPETokenizer
from utils.config import TOKENIZER_CONFIG

PROCESSED_DIR = ROOT / "data" / "processed"
TRAIN_SPLIT   = 0.9
CHUNK_SIZE    = 10_000   # characters per chunk — matches prepare.py


def strip_gutenberg(text: str) -> str:
    # Remove Project Gutenberg header/footer boilerplate
    start = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG", text, re.IGNORECASE)
    end   = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG",   text, re.IGNORECASE)
    if start:
        text = text[start.end():]
        text = text[text.find("\n"):]
    if end:
        text = text[:end.start()]
    return text.strip()


def encode_chunked(tok: ByteBPETokenizer, text: str) -> list[int]:
    chunks = [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
    ids: list[int] = []
    for chunk in tqdm(chunks, desc="    chunks", leave=False, unit="chunk"):
        ids.extend(tok.encode(chunk, add_bos=False, add_eos=False))
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Append new text files to existing corpus")
    parser.add_argument("files", nargs="+", type=Path, help="New .txt files to tokenize and append")
    parser.add_argument("--strip-gutenberg", action="store_true",
                        help="Strip Project Gutenberg header/footer boilerplate")
    parser.add_argument("--dry-run", action="store_true",
                        help="Count tokens without modifying processed files")
    args = parser.parse_args()

    tok = ByteBPETokenizer.load(TOKENIZER_CONFIG["output"])
    print(f"Tokenizer loaded: vocab_size={tok.vocab_size}", flush=True)

    # Tokenize each new file in chunks
    all_new: list[int] = []
    for path in args.files:
        if not path.exists():
            print(f"  SKIP (not found): {path}", flush=True)
            continue
        raw = path.read_text(encoding="utf-8", errors="replace")
        if args.strip_gutenberg:
            raw = strip_gutenberg(raw)
        print(f"  {path.name}: {len(raw):,} chars ({len(raw)//CHUNK_SIZE + 1} chunks)...", flush=True)
        ids = encode_chunked(tok, raw)
        all_new.extend(ids)
        print(f"    -> {len(ids):,} tokens", flush=True)

    if not all_new:
        print("No tokens produced — nothing to do.")
        return

    new_tensor = torch.tensor(all_new, dtype=torch.long)
    split_idx  = int(len(new_tensor) * TRAIN_SPLIT)
    new_train  = new_tensor[:split_idx]
    new_val    = new_tensor[split_idx:]

    print(f"\nNew tokens: {len(all_new):,}  ({split_idx:,} train / {len(new_val):,} val)")

    if args.dry_run:
        print("Dry run — processed files not modified.")
        return

    # Load existing tensors
    train_path = PROCESSED_DIR / "train.pt"
    val_path   = PROCESSED_DIR / "val.pt"
    train = torch.load(train_path, map_location="cpu", weights_only=True)
    val   = torch.load(val_path,   map_location="cpu", weights_only=True)

    print(f"Existing: {len(train):,} train / {len(val):,} val")

    train = torch.cat([train, new_train])
    val   = torch.cat([val,   new_val])

    print(f"After:    {len(train):,} train / {len(val):,} val")

    torch.save(train, train_path)
    torch.save(val,   val_path)
    print("Saved.")


if __name__ == "__main__":
    main()
