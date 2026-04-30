import sys
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from utils.config import GENERAL_CONFIG
from transformer.block import TransformerBlock


CheckResult = Tuple[str, str, str]


def run_check(name: str, check: Callable[[], str]) -> CheckResult:
    try:
        return ("PASS", name, check())
    except Exception as exc:
        return ("FAIL", name, f"{type(exc).__name__}: {exc}")


def add_result(results: List[CheckResult], name: str, check: Callable[[], str]) -> None:
    results.append(run_check(name, check))


def make_module() -> TransformerBlock:
    block = TransformerBlock()
    block.attention.return_attn_weights = False
    block.eval()
    return block


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

    def test_output_shape() -> str:
        block = make_module()
        x = make_input(batch)
        out = block(x)
        expected = (batch, seq_len, d_model)
        if out.shape != torch.Size(expected):
            raise AssertionError(f"expected {expected}, got {tuple(out.shape)}")
        return f"output.shape={tuple(out.shape)}"

    add_result(results, "output shape matches input shape (batch, seq_len, d_model)", test_output_shape)

    def test_residual_connections() -> str:
        block = make_module()
        with torch.no_grad():
            block.attention.W_o.weight.zero_()
            block.ffn.W_2.weight.zero_()
            block.ffn.W_2.bias.zero_()
        x = make_input(batch)
        out = block(x)
        if not torch.allclose(out, x, atol=1e-5):
            max_err = (out - x).abs().max().item()
            raise AssertionError(f"zeroed sublayers should give output == input, max error={max_err:.2e}")
        return f"output == input when sublayers are zeroed (max error={(out - x).abs().max().item():.2e})"

    add_result(results, "residual connections pass input through when sublayers output zero", test_residual_connections)

    def test_sublayers_contribute() -> str:
        block = make_module()
        x = make_input(batch)
        out = block(x)
        if torch.allclose(out, x, atol=1e-5):
            raise AssertionError("output is identical to input — sublayers have no effect")
        return f"output differs from input (max diff={(out - x).abs().max().item():.4f})"

    add_result(results, "sublayers contribute (output differs from input)", test_sublayers_contribute)

    def test_pre_ln_ordering() -> str:
        block = make_module()
        x = make_input(batch)

        norm1_output = {}
        attn_input   = {}

        def capture_norm1_out(module, inp, out):
            norm1_output["tensor"] = out.detach().clone()

        def capture_attn_in(module, inp, out):
            attn_input["tensor"] = inp[0].detach().clone()

        h1 = block.norm1.register_forward_hook(capture_norm1_out)
        h2 = block.attention.register_forward_hook(capture_attn_in)
        block(x)
        h1.remove()
        h2.remove()

        if not torch.allclose(norm1_output["tensor"], attn_input["tensor"], atol=1e-6):
            max_err = (norm1_output["tensor"] - attn_input["tensor"]).abs().max().item()
            raise AssertionError(f"norm1 output != attention input (max err={max_err:.2e})")
        return "norm1 output feeds directly into attention (Pre-LN confirmed)"

    add_result(results, "Pre-LN: LayerNorm applied before attention, not after", test_pre_ln_ordering)

    def test_stacked_blocks() -> str:
        block1 = make_module()
        block2 = make_module()
        x = make_input(batch)
        x.requires_grad_(True)
        out = block2(block1(x))
        expected = (batch, seq_len, d_model)
        if out.shape != torch.Size(expected):
            raise AssertionError(f"stacked shape: expected {expected}, got {tuple(out.shape)}")
        out.sum().backward()
        if x.grad is None:
            raise AssertionError("no gradient reached input")
        if x.grad.abs().max().item() == 0.0:
            raise AssertionError("gradient is all zeros")
        return f"shape preserved, gradients flow (max grad={x.grad.abs().max().item():.4f})"

    add_result(results, "two stacked blocks preserve shape and pass gradients", test_stacked_blocks)

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
