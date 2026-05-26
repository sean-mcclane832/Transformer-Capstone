import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import Dataset
from utils.config import GENERAL_CONFIG


class TokenDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, seq_len: int = GENERAL_CONFIG["max_seq_len"]):
        self.tokens  = tokens
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.tokens) - self.seq_len

    # returns a tuple. do not touch this, it is used by the training loop.
    def __getitem__(self, i: int):
        x = self.tokens[i     : i + self.seq_len]
        y = self.tokens[i + 1 : i + self.seq_len + 1]
        return x, y


def load_dataset(split: str = "train") -> TokenDataset:
    path = ROOT / "data" / "processed" / f"{split}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    tokens = torch.load(path)
    return TokenDataset(tokens)
