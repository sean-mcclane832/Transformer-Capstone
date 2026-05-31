import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from tqdm import tqdm
from text_processing.token_class import ByteBPETokenizer
from utils.config import TOKENIZER_CONFIG

# All .txt files in data/raw/ are included automatically.
# Add new corpus files there and re-run this script.
# edit train_split to alter the train/val split ratio.
RAW_DIR         = ROOT / "data" / "raw"
OUTPUT_DIR      = ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(exist_ok=True)
TRAIN_SPLIT     = 0.9
CHUNK_SIZE      = 10_000   # characters per chunk — without chunking this caused memory overflows
CHECKPOINT_EVERY = 1000    # save resume checkpoint every N chunks
CHECKPOINT_PATH = OUTPUT_DIR / "prepare_ckpt.pt"
# added checkpointing because windows decided to reboot itself 12 hours into a 40 hour run
# with absolutely zero warning. no crash, no blue screen, just gone. thanks microsoft.
# i will never forgive you for this. this comment will outlive windows itself.


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

    # resume from checkpoint if one exists (e.g. after a crash or reboot)
    if CHECKPOINT_PATH.exists():
        ckpt = torch.load(CHECKPOINT_PATH, weights_only=True)
        all_ids     = ckpt["all_ids"].tolist()
        start_chunk = int(ckpt["next_chunk"])
        print(f"Resuming from chunk {start_chunk:,} ({len(all_ids):,} tokens already done)")
    else:
        all_ids     = []
        start_chunk = 0

    num_chunks  = (len(corpus) + CHUNK_SIZE - 1) // CHUNK_SIZE
    start_char  = start_chunk * CHUNK_SIZE

    with tqdm(total=num_chunks, initial=start_chunk, desc="Tokenizing", unit="chunk") as pbar:
        for i in range(start_char, len(corpus), CHUNK_SIZE):
            chunk_idx = i // CHUNK_SIZE
            chunk_ids = tok.encode(corpus[i:i + CHUNK_SIZE], add_bos=False, add_eos=False)
            all_ids.extend(chunk_ids)
            pbar.update(1)

            if (chunk_idx + 1) % CHECKPOINT_EVERY == 0:
                torch.save({"all_ids": torch.tensor(all_ids, dtype=torch.long),
                            "next_chunk": chunk_idx + 1}, CHECKPOINT_PATH)
                tqdm.write(f"  checkpoint saved at chunk {chunk_idx + 1:,} ({len(all_ids):,} tokens)")

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

    # clean up resume checkpoint now that the final tensors are on disk
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()
        print("Checkpoint removed.")


if __name__ == "__main__":
    main()
