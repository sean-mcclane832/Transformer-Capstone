import sys
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from utils.config import GENERAL_CONFIG
from data.dataset import TokenDataset, load_dataset


CheckResult = Tuple[str, str, str]


def run_check(name: str, check: Callable[[], str]) -> CheckResult:
    try:
        return ("PASS", name, check())
    except Exception as exc:
        return ("FAIL", name, f"{type(exc).__name__}: {exc}")


def add_result(results: List[CheckResult], name: str, check: Callable[[], str]) -> None:
    results.append(run_check(name, check))


def make_dataset(length: int = 500) -> TokenDataset:
    torch.manual_seed(GENERAL_CONFIG["seed"])
    tokens = torch.randint(0, GENERAL_CONFIG["vocab_size"], (length,), dtype=torch.long)
    return TokenDataset(tokens)


def main() -> None:
    results: List[CheckResult] = []

    seq_len = GENERAL_CONFIG["max_seq_len"]
    print(f"Config: seq_len={seq_len}, vocab_size={GENERAL_CONFIG['vocab_size']}")
    print()

    # ------------------------------------------------------------------ #
    # 1. __len__                                                           #
    # ------------------------------------------------------------------ #
    def test_length() -> str:
        n_tokens = 500
        ds = make_dataset(n_tokens)
        expected = n_tokens - seq_len
        if len(ds) != expected:
            raise AssertionError(f"expected len={expected}, got {len(ds)}")
        return f"len={len(ds)} (n_tokens={n_tokens} - seq_len={seq_len})"

    add_result(results, "__len__ = n_tokens - seq_len", test_length)

    # ------------------------------------------------------------------ #
    # 2. x and y shapes                                                    #
    # ------------------------------------------------------------------ #
    def test_item_shapes() -> str:
        ds = make_dataset()
        x, y = ds[0]
        if x.shape != torch.Size([seq_len]):
            raise AssertionError(f"x shape: expected ({seq_len},), got {tuple(x.shape)}")
        if y.shape != torch.Size([seq_len]):
            raise AssertionError(f"y shape: expected ({seq_len},), got {tuple(y.shape)}")
        return f"x.shape={tuple(x.shape)}, y.shape={tuple(y.shape)}"

    add_result(results, "x and y shapes are (seq_len,)", test_item_shapes)

    # ------------------------------------------------------------------ #
    # 3. y is x shifted by 1                                              #
    # ------------------------------------------------------------------ #
    def test_shift() -> str:
        ds = make_dataset()
        x, y = ds[0]
        x1, _ = ds[1]
        # y[:-1] should equal x[1:], and y[-1] should equal x1[0]
        if not torch.equal(y[:-1], x[1:]):
            raise AssertionError("y[:-1] != x[1:] — shift-by-one violated")
        if y[-1].item() != x1[0].item():
            raise AssertionError(f"y[-1]={y[-1].item()} != next window x[0]={x1[0].item()}")
        return "y == x shifted left by one position"

    add_result(results, "y is x shifted by 1 (next-token targets)", test_shift)

    # ------------------------------------------------------------------ #
    # 4. DataLoader batching                                               #
    # ------------------------------------------------------------------ #
    def test_dataloader() -> str:
        ds = make_dataset()
        batch_size = 4
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        x_batch, y_batch = next(iter(loader))
        expected = (batch_size, seq_len)
        if x_batch.shape != torch.Size(expected):
            raise AssertionError(f"batch x shape: expected {expected}, got {tuple(x_batch.shape)}")
        if y_batch.shape != torch.Size(expected):
            raise AssertionError(f"batch y shape: expected {expected}, got {tuple(y_batch.shape)}")
        return f"batch shape: {tuple(x_batch.shape)}"

    add_result(results, "DataLoader produces batches of shape (batch, seq_len)", test_dataloader)

    # ------------------------------------------------------------------ #
    # 5. Token IDs are valid (within vocab range)                         #
    # ------------------------------------------------------------------ #
    def test_token_range() -> str:
        ds = make_dataset()
        x, y = ds[0]
        vocab_size = GENERAL_CONFIG["vocab_size"]
        if x.min().item() < 0 or x.max().item() >= vocab_size:
            raise AssertionError(f"x tokens out of range [0, {vocab_size}): min={x.min()}, max={x.max()}")
        return f"all tokens in [0, {vocab_size})"

    add_result(results, "token IDs are within vocab range", test_token_range)

    # ------------------------------------------------------------------ #
    # 6. load_dataset reads from disk (skipped if file absent)           #
    # ------------------------------------------------------------------ #
    def test_load_dataset() -> str:
        path = ROOT / "data" / "train.pt"
        if not path.exists():
            raise AssertionError(f"train.pt not found at {path} — run data/prepare.py first")
        ds = load_dataset("train")
        if len(ds) == 0:
            raise AssertionError("loaded dataset is empty")
        return f"loaded train split: {len(ds)} windows"

    add_result(results, "load_dataset('train') loads from disk", test_load_dataset)

    # ------------------------------------------------------------------ #
    # Print results                                                        #
    # ------------------------------------------------------------------ #
    print("Results:")
    passed = failed = 0
    for status, name, detail in results:
        print(f"  [{status}] {name}: {detail}")
        if status == "PASS":
            passed += 1
        else:
            failed += 1

    print()
    print(f"Summary: {passed} passed, {failed} failed")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
