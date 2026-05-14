import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from text_processing.token_class import ByteBPETokenizer
from utils.config import TOKENIZER_CONFIG

# All .txt files in data/raw/ are included automatically.
# Add new corpus files there and re-run this script.
# edit train_split to alter the train/val split ratio.
RAW_DIR    = ROOT / "data" / "raw"
OUTPUT_DIR = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)
TRAIN_SPLIT = 0.9


def main() -> None:
    tok = ByteBPETokenizer.load(TOKENIZER_CONFIG["output"])
    print(f"Tokenizer loaded: vocab_size={tok.vocab_size}")

    corpus_files = sorted(RAW_DIR.glob("*.txt"))
    if not corpus_files:
        raise FileNotFoundError(f"No .txt files found in {RAW_DIR}")

    print(f"Corpus files ({len(corpus_files)}):")
    corpus = ""
    for path in corpus_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        print(f"  {path.name}: {len(text):,} chars")
        corpus += text

    print(f"Total corpus: {len(corpus):,} chars")

    # Tokenize in chunks
    # without the chunks this caused memory overflows and wouldnt work
    CHUNK_SIZE = 10_000  # characters per chunk
    all_ids = []
    num_chunks = (len(corpus) + CHUNK_SIZE - 1) // CHUNK_SIZE
    for i in range(0, len(corpus), CHUNK_SIZE):
        chunk_ids = tok.encode(corpus[i:i + CHUNK_SIZE], add_bos=False, add_eos=False)
        all_ids.extend(chunk_ids)
        print(f"{i // CHUNK_SIZE + 1}/{num_chunks}")

    print(f"Total tokens: {len(all_ids):,}")

    tokens = torch.tensor(all_ids, dtype=torch.long)

    # --- 90/10 train/val split ---
    # DO NOT SHUFFLE
    n_train = int(len(tokens) * TRAIN_SPLIT)
    train_tokens = tokens[:n_train]
    val_tokens   = tokens[n_train:]

    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens:   {len(val_tokens):,}")

    # save token tensors to disk. these are loaded by the Dataset class during training.
    train_path = OUTPUT_DIR / "train.pt"
    val_path   = OUTPUT_DIR / "val.pt"

    torch.save(train_tokens, train_path)
    torch.save(val_tokens,   val_path)

    print(f"Saved: {train_path}")
    print(f"Saved: {val_path}")


if __name__ == "__main__":
    main()
