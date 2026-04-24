#test script made by Claude Code, works as intended.
import sys
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from utils.config import GENERAL_CONFIG
from attention.multi_head import MultiHeadAttention


CheckResult = Tuple[str, str, str]


def run_check(name: str, check: Callable[[], str]) -> CheckResult:
    try:
        return ("PASS", name, check())
    except Exception as exc:
        return ("FAIL", name, f"{type(exc).__name__}: {exc}")


def add_result(results: List[CheckResult], name: str, check: Callable[[], str]) -> None:
    results.append(run_check(name, check))


def make_module() -> MultiHeadAttention:
    """Return an eval-mode MultiHeadAttention with weights always returned."""
    mha = MultiHeadAttention()
    mha.return_attn_weights = True
    mha.eval()
    return mha


def make_input(batch: int = 2) -> torch.Tensor:
    seq_len = GENERAL_CONFIG["max_seq_len"]
    d_model = GENERAL_CONFIG["d_model"]
    torch.manual_seed(GENERAL_CONFIG["seed"])
    return torch.randn(batch, seq_len, d_model)


def main() -> None:
    results: List[CheckResult] = []

    batch   = 2
    seq_len = GENERAL_CONFIG["max_seq_len"]
    d_model = GENERAL_CONFIG["d_model"]
    n_heads = GENERAL_CONFIG["n_heads"]

    print(f"Config: batch={batch}, seq_len={seq_len}, d_model={d_model}, n_heads={n_heads}")
    print()

    # ------------------------------------------------------------------ #
    # 1. Output shape                                                      #
    # ------------------------------------------------------------------ #
    def test_output_shape() -> str:
        mha = make_module()
        x = make_input(batch)
        output, _ = mha(x)
        expected = (batch, seq_len, d_model)
        if output.shape != torch.Size(expected):
            raise AssertionError(f"expected {expected}, got {tuple(output.shape)}")
        return f"output.shape={tuple(output.shape)}"

    add_result(results, "output shape is (batch, seq_len, d_model)", test_output_shape)

    # ------------------------------------------------------------------ #
    # 2. Attention weights shape                                           #
    # ------------------------------------------------------------------ #
    def test_weights_shape() -> str:
        mha = make_module()
        x = make_input(batch)
        _, weights = mha(x)
        expected = (batch, n_heads, seq_len, seq_len)
        if weights.shape != torch.Size(expected):
            raise AssertionError(f"expected {expected}, got {tuple(weights.shape)}")
        return f"weights.shape={tuple(weights.shape)}"

    add_result(results, "weights shape is (batch, n_heads, seq_len, seq_len)", test_weights_shape)

    # ------------------------------------------------------------------ #
    # 3. Weights sum to 1 along last dim (softmax rows)                   #
    # ------------------------------------------------------------------ #
    def test_weights_sum_to_one() -> str:
        mha = make_module()
        x = make_input(batch)
        _, weights = mha(x)
        row_sums = weights.sum(dim=-1)  # (batch, n_heads, seq_len)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            max_err = (row_sums - 1.0).abs().max().item()
            raise AssertionError(f"rows do not sum to 1 — max deviation: {max_err:.2e}")
        max_err = (row_sums - 1.0).abs().max().item()
        return f"all rows sum to 1 (max deviation={max_err:.2e})"

    add_result(results, "attention weights sum to 1 per row", test_weights_sum_to_one)

    # ------------------------------------------------------------------ #
    # 4. Causal mask — upper triangle must be zero                        #
    # ------------------------------------------------------------------ #
    def test_causal_mask_enforced() -> str:
        mha = make_module()
        x = make_input(batch)
        _, weights = mha(x)

        # Positions j > i should be exactly 0 after softmax(-inf)
        # triu(..., diagonal=1) selects everything strictly above the diagonal
        upper = torch.triu(weights, diagonal=1)
        max_val = upper.abs().max().item()
        if max_val > 1e-7:
            bad = (upper.abs() > 1e-7).nonzero(as_tuple=False)[0].tolist()
            raise AssertionError(
                f"future token attended at index {bad} — max upper-triangle value: {max_val:.2e}"
            )
        return f"upper triangle is zero (max={max_val:.2e})"

    add_result(results, "causal mask zeroes out future positions", test_causal_mask_enforced)

    # ------------------------------------------------------------------ #
    # Print results                                                        #
    # ------------------------------------------------------------------ #
    print("Results:")
    passed = failed = skipped = 0
    for status, name, detail in results:
        print(f"  [{status}] {name}: {detail}")
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        else:
            skipped += 1

    print()
    print(f"Summary: {passed} passed, {failed} failed, {skipped} skipped")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
