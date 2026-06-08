"""
test_gqa.py — Unit tests for Grouped-Query Attention

Tests that GQA with n_kv_heads < n_heads produces correct shapes, valid attention weights,
and different output than standard MHA (proving KV sharing actually changes the computation).
Also verifies the n_groups=1 (MQA) and n_groups=n_heads (full MHA fallback) edge cases.
"""

import sys
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import utils.config as _cfg
from attention.multi_head import MultiHeadAttention

CheckResult = Tuple[str, str, str]


def run_check(name: str, check: Callable[[], str]) -> CheckResult:
    try:
        return ("PASS", name, check())
    except Exception as exc:
        return ("FAIL", name, f"{type(exc).__name__}: {exc}")


def make_mha(n_kv_heads=None) -> MultiHeadAttention:
    """Instantiate MHA with a patched config so we can test different n_kv_heads values."""
    original = dict(_cfg.GENERAL_CONFIG)
    _cfg.GENERAL_CONFIG["n_kv_heads"] = n_kv_heads
    _cfg.GENERAL_CONFIG["use_rope"]   = False   # isolate GQA from RoPE in unit tests
    try:
        mha = MultiHeadAttention()
        mha.return_attn_weights = True
        mha.eval()
    finally:
        _cfg.GENERAL_CONFIG.clear()
        _cfg.GENERAL_CONFIG.update(original)
    return mha


def make_input(batch=2):
    seq_len = _cfg.GENERAL_CONFIG["max_seq_len"]
    d_model = _cfg.GENERAL_CONFIG["d_model"]
    torch.manual_seed(_cfg.GENERAL_CONFIG["seed"])
    return torch.randn(batch, seq_len, d_model)


def main():
    results: List[CheckResult] = []
    batch   = 2
    n_heads = _cfg.GENERAL_CONFIG["n_heads"]
    d_model = _cfg.GENERAL_CONFIG["d_model"]
    seq_len = _cfg.GENERAL_CONFIG["max_seq_len"]

    print(f"Config: batch={batch}, seq_len={seq_len}, d_model={d_model}, n_heads={n_heads}")
    print()

    # ── 1. MHA fallback (n_kv_heads=None → standard MHA) ─────────────────

    def test_mha_fallback_output_shape():
        mha = make_mha(n_kv_heads=None)
        x = make_input(batch)
        out, _ = mha(x)
        expected = (batch, seq_len, d_model)
        assert out.shape == torch.Size(expected), f"got {tuple(out.shape)}"
        assert mha.n_kv_heads == n_heads, f"n_kv_heads should default to n_heads, got {mha.n_kv_heads}"
        return f"output.shape={tuple(out.shape)}, n_kv_heads={mha.n_kv_heads}"

    results.append(run_check("MHA fallback: output shape and n_kv_heads=n_heads", test_mha_fallback_output_shape))

    # ── 2. GQA output shape (n_kv_heads=n_heads//3) ──────────────────────

    n_kv = n_heads // 3  # e.g. 4 for small (12 heads → 3 groups of 4 queries per KV)

    def test_gqa_output_shape():
        mha = make_mha(n_kv_heads=n_kv)
        x = make_input(batch)
        out, _ = mha(x)
        expected = (batch, seq_len, d_model)
        assert out.shape == torch.Size(expected), f"got {tuple(out.shape)}"
        return f"output.shape={tuple(out.shape)}, n_kv_heads={mha.n_kv_heads}, n_groups={mha.n_groups}"

    results.append(run_check(f"GQA output shape (n_kv_heads={n_kv})", test_gqa_output_shape))

    # ── 3. Attention weights shape — should still be (batch, n_heads, seq, seq) ─

    def test_gqa_weights_shape():
        mha = make_mha(n_kv_heads=n_kv)
        x = make_input(batch)
        _, weights = mha(x)
        expected = (batch, n_heads, seq_len, seq_len)
        assert weights.shape == torch.Size(expected), f"got {tuple(weights.shape)}"
        return f"weights.shape={tuple(weights.shape)}"

    results.append(run_check("GQA weights shape is (batch, n_heads, seq, seq)", test_gqa_weights_shape))

    # ── 4. Attention weights sum to 1 (valid probability rows) ───────────

    def test_gqa_weights_sum_to_one():
        mha = make_mha(n_kv_heads=n_kv)
        x = make_input(batch)
        _, weights = mha(x)
        row_sums = weights.sum(dim=-1)
        max_err = (row_sums - 1.0).abs().max().item()
        assert max_err < 1e-5, f"rows don't sum to 1 — max err {max_err:.2e}"
        return f"max deviation from 1.0: {max_err:.2e}"

    results.append(run_check("GQA weights sum to 1 per row", test_gqa_weights_sum_to_one))

    # ── 5. Causal mask enforced ───────────────────────────────────────────

    def test_gqa_causal_mask():
        mha = make_mha(n_kv_heads=n_kv)
        x = make_input(batch)
        _, weights = mha(x)
        upper = torch.triu(weights, diagonal=1)
        max_val = upper.abs().max().item()
        assert max_val < 1e-7, f"future token attended — max upper-tri: {max_val:.2e}"
        return f"upper triangle is zero (max={max_val:.2e})"

    results.append(run_check("GQA causal mask enforced", test_gqa_causal_mask))

    # ── 6. GQA W_k/W_v are smaller than MHA ──────────────────────────────

    def test_gqa_projection_sizes():
        mha_full = make_mha(n_kv_heads=None)
        mha_gqa  = make_mha(n_kv_heads=n_kv)
        d_k = d_model // n_heads
        expected_kv_out = n_kv * d_k
        full_kv_out     = n_heads * d_k  # = d_model
        gqa_k_out = mha_gqa.projections.W_k.out_features
        assert gqa_k_out == expected_kv_out, f"W_k out_features: expected {expected_kv_out}, got {gqa_k_out}"
        assert mha_full.projections.W_k.out_features == full_kv_out
        reduction = full_kv_out / gqa_k_out
        return f"W_k: {full_kv_out}->{gqa_k_out} ({reduction:.1f}x reduction)"

    results.append(run_check(f"GQA W_k/W_v are {n_heads//n_kv}× smaller than MHA", test_gqa_projection_sizes))

    # ── 7. GQA and MHA produce different outputs (KV sharing has effect) ─

    def test_gqa_differs_from_mha():
        torch.manual_seed(0)
        mha_full = make_mha(n_kv_heads=None)
        torch.manual_seed(0)
        mha_gqa  = make_mha(n_kv_heads=n_kv)
        x = make_input(batch)
        with torch.no_grad():
            out_full, _ = mha_full(x)
            out_gqa,  _ = mha_gqa(x)
        diff = (out_full - out_gqa).abs().max().item()
        assert diff > 1e-4, f"GQA and MHA outputs are identical — GQA may not be active (diff={diff:.2e})"
        return f"max output diff between MHA and GQA: {diff:.4f}"

    results.append(run_check("GQA output differs from full MHA (KV sharing is active)", test_gqa_differs_from_mha))

    # ── 8. MQA edge case (n_kv_heads=1) ──────────────────────────────────

    def test_mqa_edge_case():
        mha = make_mha(n_kv_heads=1)
        x = make_input(batch)
        out, weights = mha(x)
        assert out.shape == torch.Size([batch, seq_len, d_model])
        assert weights.shape == torch.Size([batch, n_heads, seq_len, seq_len])
        assert mha.n_groups == n_heads
        return f"MQA: n_kv_heads=1, n_groups={mha.n_groups}, output {tuple(out.shape)}"

    results.append(run_check("MQA edge case (n_kv_heads=1, all queries share one KV)", test_mqa_edge_case))

    # ── Print results ─────────────────────────────────────────────────────

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
