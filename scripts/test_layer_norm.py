import sys
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from utils.config import GENERAL_CONFIG
from transformer.layer_norm import LayerNorm


CheckResult = Tuple[str, str, str]


def run_check(name: str, check: Callable[[], str]) -> CheckResult:
    try:
        return ("PASS", name, check())
    except Exception as exc:
        return ("FAIL", name, f"{type(exc).__name__}: {exc}")


def add_result(results: List[CheckResult], name: str, check: Callable[[], str]) -> None:
    results.append(run_check(name, check))


def main() -> None:
    results: List[CheckResult] = []

    batch   = 2
    seq_len = GENERAL_CONFIG["max_seq_len"]
    d_model = GENERAL_CONFIG["d_model"]

    print(f"Config: batch={batch}, seq_len={seq_len}, d_model={d_model}")
    print()

    def test_output_shape() -> str:
        ln = LayerNorm(d_model)
        torch.manual_seed(GENERAL_CONFIG["seed"])
        x = torch.randn(batch, seq_len, d_model)
        out = ln(x)
        if out.shape != x.shape:
            raise AssertionError(f"expected {tuple(x.shape)}, got {tuple(out.shape)}")
        return f"output.shape={tuple(out.shape)}"

    add_result(results, "output shape matches input shape", test_output_shape)

    def test_normalized_statistics() -> str:
        ln = LayerNorm(d_model)
        ln.eval()
        torch.manual_seed(GENERAL_CONFIG["seed"])
        x = torch.randn(batch, seq_len, d_model)
        out = ln(x)
        mean = out.mean(dim=-1)
        std  = out.std(dim=-1, unbiased=False)
        max_mean_err = mean.abs().max().item()
        max_std_err  = (std - 1.0).abs().max().item()
        if max_mean_err > 1e-5:
            raise AssertionError(f"output mean is not ~0 — max deviation={max_mean_err:.2e}")
        if max_std_err > 1e-5:
            raise AssertionError(f"output std is not ~1 — max deviation={max_std_err:.2e}")
        return f"mean~0 (max err={max_mean_err:.2e}), std~1 (max err={max_std_err:.2e})"

    add_result(results, "output has mean~0 and std~1 per position (default gamma/beta)", test_normalized_statistics)

    def test_numeric_correctness() -> str:
        ln = LayerNorm(4)
        with torch.no_grad():
            ln.gamma.fill_(1.0)
            ln.beta.fill_(0.0)

        # x = [1, 2, 3, 4], mean=2.5, var=1.25
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]]])
        out = ln(x)
        eps = 1e-5
        expected = torch.tensor(
            [[(v - 2.5) / (1.25 + eps) ** 0.5 for v in [1.0, 2.0, 3.0, 4.0]]]
        ).unsqueeze(0)

        if not torch.allclose(out, expected, atol=1e-4):
            raise AssertionError(f"numeric mismatch — max error={(out - expected).abs().max().item():.2e}")
        return f"output matches hand-calculated values (max error={(out - expected).abs().max().item():.2e})"

    add_result(results, "numeric correctness against hand-calculated values", test_numeric_correctness)

    def test_gamma_scales() -> str:
        scale = 3.0
        ln_base   = LayerNorm(d_model)
        ln_scaled = LayerNorm(d_model)
        with torch.no_grad():
            ln_base.gamma.fill_(1.0);  ln_base.beta.fill_(0.0)
            ln_scaled.gamma.fill_(scale); ln_scaled.beta.fill_(0.0)
        torch.manual_seed(0)
        x = torch.randn(batch, seq_len, d_model)
        expected = scale * ln_base(x)
        out = ln_scaled(x)
        if not torch.allclose(out, expected, atol=1e-5):
            raise AssertionError(f"gamma={scale} did not scale output — max err={(out - expected).abs().max().item():.2e}")
        return f"gamma={scale} multiplies output by {scale}"

    add_result(results, "gamma parameter scales the normalized output", test_gamma_scales)

    def test_beta_shifts() -> str:
        shift = 5.0
        ln_base    = LayerNorm(d_model)
        ln_shifted = LayerNorm(d_model)
        with torch.no_grad():
            ln_base.gamma.fill_(1.0);    ln_base.beta.fill_(0.0)
            ln_shifted.gamma.fill_(1.0); ln_shifted.beta.fill_(shift)
        torch.manual_seed(0)
        x = torch.randn(batch, seq_len, d_model)
        expected = ln_base(x) + shift
        out = ln_shifted(x)
        if not torch.allclose(out, expected, atol=1e-5):
            raise AssertionError(f"beta={shift} did not shift output — max err={(out - expected).abs().max().item():.2e}")
        return f"beta={shift} adds {shift} to every output element"

    add_result(results, "beta parameter shifts the normalized output", test_beta_shifts)

    def test_learnable_parameters() -> str:
        ln = LayerNorm(d_model)
        param_names = [name for name, _ in ln.named_parameters()]
        if "gamma" not in param_names:
            raise AssertionError(f"gamma not in parameters — got {param_names}")
        if "beta" not in param_names:
            raise AssertionError(f"beta not in parameters — got {param_names}")
        total_params = sum(p.numel() for p in ln.parameters())
        if total_params != 2 * d_model:
            raise AssertionError(f"expected {2 * d_model} parameters, got {total_params}")
        return f"gamma and beta registered, total params={total_params} (2 * d_model={2*d_model})"

    add_result(results, "gamma and beta are registered as learnable parameters", test_learnable_parameters)

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
