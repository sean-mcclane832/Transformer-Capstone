import sys
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from utils.config import GENERAL_CONFIG
from attention.projections import AttentionProjections


CheckResult = Tuple[str, str, str]


def run_check(name: str, check: Callable[[], str]) -> CheckResult:
    try:
        return ("PASS", name, check())
    except Exception as exc:
        return ("FAIL", name, f"{type(exc).__name__}: {exc}")


def add_result(results: List[CheckResult], name: str, check: Callable[[], str]) -> None:
    results.append(run_check(name, check))


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

    print(f"Config: batch={batch}, seq_len={seq_len}, d_model={d_model}")
    print()

    # ------------------------------------------------------------------ #
    # 1. Output shapes                                                     #
    # ------------------------------------------------------------------ #
    def test_output_shapes() -> str:
        proj = AttentionProjections(d_model)
        proj.eval()
        x = make_input(batch)
        Q, K, V = proj(x)
        expected = torch.Size([batch, seq_len, d_model])
        for name, t in [("Q", Q), ("K", K), ("V", V)]:
            if t.shape != expected:
                raise AssertionError(f"{name}: expected {tuple(expected)}, got {tuple(t.shape)}")
        return f"Q={tuple(Q.shape)}, K={tuple(K.shape)}, V={tuple(V.shape)}"

    add_result(results, "Q, K, V shapes are (batch, seq_len, d_model)", test_output_shapes)

    # ------------------------------------------------------------------ #
    # 2. Q, K, V are independent projections                              #
    #    Same input through different weight matrices must give            #
    #    different outputs.                                                #
    # ------------------------------------------------------------------ #
    def test_projections_are_independent() -> str:
        proj = AttentionProjections(d_model)
        proj.eval()
        x = make_input(batch)
        Q, K, V = proj(x)
        if torch.allclose(Q, K, atol=1e-6):
            raise AssertionError("Q and K are identical — W_q and W_k may share weights")
        if torch.allclose(Q, V, atol=1e-6):
            raise AssertionError("Q and V are identical — W_q and W_v may share weights")
        if torch.allclose(K, V, atol=1e-6):
            raise AssertionError("K and V are identical — W_k and W_v may share weights")
        return "Q, K, V are all distinct"

    add_result(results, "Q, K, V are independent (different weight matrices)", test_projections_are_independent)

    # ------------------------------------------------------------------ #
    # 3. Numeric correctness — known weights, known input                 #
    #    With W_q set to identity, Q must equal x exactly.               #
    # ------------------------------------------------------------------ #
    def test_numeric_correctness() -> str:
        proj = AttentionProjections(d_model)
        proj.eval()

        # Set W_q to identity, zero bias — output should equal input
        with torch.no_grad():
            proj.W_q.weight.copy_(torch.eye(d_model))
            proj.W_q.bias.zero_()

        x = make_input(batch)
        Q, _, _ = proj(x)

        if not torch.allclose(Q, x, atol=1e-5):
            max_err = (Q - x).abs().max().item()
            raise AssertionError(f"W_q=I produced Q != x — max error={max_err:.2e}")
        return f"W_q=Identity => Q==x (max error={( Q - x).abs().max().item():.2e})"

    add_result(results, "W_q=Identity with zero bias produces Q equal to input", test_numeric_correctness)

    # ------------------------------------------------------------------ #
    # 4. Linearity — W_q(a + b) == W_q(a) + W_q(b)                      #
    #    Linear layers must satisfy superposition.                        #
    # ------------------------------------------------------------------ #
    def test_linearity() -> str:
        proj = AttentionProjections(d_model)
        proj.eval()

        torch.manual_seed(0)
        a = torch.randn(1, seq_len, d_model)
        torch.manual_seed(1)
        b = torch.randn(1, seq_len, d_model)

        # Zero bias so superposition holds cleanly
        with torch.no_grad():
            proj.W_q.bias.zero_()

        Qa, _, _ = proj(a)
        Qb, _, _ = proj(b)
        Qab, _, _ = proj(a + b)

        if not torch.allclose(Qab, Qa + Qb, atol=1e-5):
            max_err = (Qab - (Qa + Qb)).abs().max().item()
            raise AssertionError(f"W_q is not linear — max superposition error={max_err:.2e}")
        max_err = (Qab - (Qa + Qb)).abs().max().item()
        return f"W_q(a+b) == W_q(a)+W_q(b) (max error={max_err:.2e})"

    add_result(results, "projections satisfy linearity (W_q(a+b) == W_q(a) + W_q(b))", test_linearity)

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
