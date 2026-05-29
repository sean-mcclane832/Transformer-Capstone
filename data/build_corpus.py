"""
data/build_corpus.py

Downloads and exports all corpus sources to data/raw/ as plain .txt files.
Run once before prepare.py.
"""

from datasets import load_dataset
from pathlib import Path

RAW_DIR = Path(__file__).parent / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)


def export_wikitext():
    print("Downloading WikiText-103...")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    out = RAW_DIR / "wikitext103.txt"
    with open(out, "w", encoding="utf-8") as f:
        for row in ds:
            text = row["text"].strip()
            if text:
                f.write(text + "\n")
    print(f"  → {out} written")


def export_pg19(max_books: int = 200):
    print(f"Downloading PG-19 (first {max_books} books)...")
    ds = load_dataset("emozilla/pg19", split="train", streaming=True)
    out = RAW_DIR / "pg19.txt"
    with open(out, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if i >= max_books:
                break
            f.write(row["text"].strip() + "\n\n")
    print(f"  → {out} written")


def export_openwebtext(max_docs: int = 50_000):
    print(f"Downloading OpenWebText (first {max_docs} documents)...")
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    out = RAW_DIR / "openwebtext.txt"
    with open(out, "w", encoding="utf-8") as f:
        for i, row in enumerate(ds):
            if i >= max_docs:
                break
            text = row["text"].strip()
            if text:
                f.write(text + "\n\n")
    print(f"  → {out} written")


if __name__ == "__main__":
    export_wikitext()
    export_pg19(max_books=200)
    export_openwebtext(max_docs=50_000)
    print("\nDone. Run prepare.py next.")