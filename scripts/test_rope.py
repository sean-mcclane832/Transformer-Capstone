#test script for RoPE — attention/rope.py and use_rope flag in MultiHeadAttention
import sys
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from utils.config import GENERAL_CONFIG
from attention.rope import RotaryEmbedding, apply_rotary, _rotate_half
from attention.multi_head import MultiHeadAttention


CheckResult = Tuple[str, str, str]


def run_check(name: str, check: Callable[[], str]) -> CheckResult:
    try:
        return ("PASS", name, check())
    except Exception as exc:
        return ("FAIL", name, f"{type(exc).__name__}: {exc}")


def add_result(results: List[CheckResult], name: str, check: Callable[[], str]) -> None:
    results.append(run_check(name, check))


def make_rope(d_k: int = 16, max_seq_len: int = 32) -> RotaryEmbedding:
    return RotaryEmbedding(d_k, max_seq_len).eval()


def make_mha_rope() -> MultiHeadAttention:
    # temporarily enable rope so the module instantiates with it
    original = GENERAL_CONFIG.get("use_rope", False)
    GENERAL_CONFIG["use_rope"] = True
    mha = MultiHeadAttention()
    mha.return_attn_weights = True
    mha.eval()
    GENERAL_CONFIG["use_rope"] = original
    return mha


def main() -> None:
    results: List[CheckResult] = []

    d_k      = 16
    seq_len  = 32
    batch    = 2
    n_heads  = 4
    torch.manual_seed(42)

    print(f"Config: batch={batch}, seq_len={seq_len}, d_k={d_k}, n_heads={n_heads}")
    print()

    # ------------------------------------------------------------------ #
    # 1. apply_rotary preserves shape                                      #
    # ------------------------------------------------------------------ #
    def test_shape() -> str:
        rope = make_rope(d_k, seq_len)
        x    = torch.randn(batch, n_heads, seq_len, d_k)
        cos, sin = rope(seq_len)
        out  = apply_rotary(x, cos, sin)
        if out.shape != x.shape:
            raise AssertionError(f"expected {x.shape}, got {out.shape}")
        return f"out.shape={tuple(out.shape)}"

    add_result(results, "apply_rotary preserves input shape", test_shape)

    # ------------------------------------------------------------------ #
    # 2. rotation is position-dependent                                    #
    # ------------------------------------------------------------------ #
    def test_position_dependent() -> str:
        rope = make_rope(d_k, seq_len)
        x    = torch.randn(1, 1, seq_len, d_k)
        cos, sin = rope(seq_len)
        out  = apply_rotary(x, cos, sin)
        # if rotation were position-independent all rows would be identical
        # check that at least two positions differ
        if torch.allclose(out[0, 0, 0], out[0, 0, 1], atol=1e-5):
            raise AssertionError("position 0 and position 1 produced identical output — rotation is not position-dependent")
        delta = (out[0, 0, 0] - out[0, 0, 1]).abs().max().item()
        return f"positions differ by up to {delta:.4f}"

    add_result(results, "rotation varies by sequence position", test_position_dependent)

    # ------------------------------------------------------------------ #
    # 3. rotation preserves vector norm                                    #
    # ------------------------------------------------------------------ #
    def test_norm_preserved() -> str:
        rope = make_rope(d_k, seq_len)
        x    = torch.randn(batch, n_heads, seq_len, d_k)
        cos, sin = rope(seq_len)
        out  = apply_rotary(x, cos, sin)
        norm_in  = x.norm(dim=-1)
        norm_out = out.norm(dim=-1)
        if not torch.allclose(norm_in, norm_out, atol=1e-5):
            max_err = (norm_in - norm_out).abs().max().item()
            raise AssertionError(f"norms changed by up to {max_err:.2e} — rotation should be isometric")
        max_err = (norm_in - norm_out).abs().max().item()
        return f"norms preserved (max deviation={max_err:.2e})"

    add_result(results, "rotation preserves vector norm (isometric)", test_norm_preserved)

    # ------------------------------------------------------------------ #
    # 4. rotate_half is its own inverse when applied twice                 #
    # ------------------------------------------------------------------ #
    def test_rotate_half_involution() -> str:
        x   = torch.randn(2, 4, 8)
        out = _rotate_half(_rotate_half(x))
        # rotate_half twice should return -x  (rotate 180°)
        if not torch.allclose(out, -x, atol=1e-6):
            raise AssertionError("_rotate_half applied twice should give -x")
        return "_rotate_half(_rotate_half(x)) == -x"

    add_result(results, "_rotate_half twice gives -x (180° rotation)", test_rotate_half_involution)

    # ------------------------------------------------------------------ #
    # 5. MultiHeadAttention with use_rope=True — correct output shape      #
    # ------------------------------------------------------------------ #
    def test_mha_rope_output_shape() -> str:
        mha     = make_mha_rope()
        d_model = GENERAL_CONFIG["d_model"]
        sl      = GENERAL_CONFIG["max_seq_len"]
        x       = torch.randn(batch, sl, d_model)
        output, _ = mha(x)
        expected  = (batch, sl, d_model)
        if output.shape != torch.Size(expected):
            raise AssertionError(f"expected {expected}, got {tuple(output.shape)}")
        return f"output.shape={tuple(output.shape)}"

    add_result(results, "MHA with use_rope=True output shape is (batch, seq_len, d_model)", test_mha_rope_output_shape)

    # ------------------------------------------------------------------ #
    # 6. Causal mask still enforced with RoPE on                          #
    # ------------------------------------------------------------------ #
    def test_causal_mask_with_rope() -> str:
        mha = make_mha_rope()
        d_model = GENERAL_CONFIG["d_model"]
        sl      = GENERAL_CONFIG["max_seq_len"]
        x       = torch.randn(batch, sl, d_model)
        _, weights = mha(x)
        upper   = torch.triu(weights, diagonal=1)
        max_val = upper.abs().max().item()
        if max_val > 1e-7:
            raise AssertionError(f"future token attended — max upper-triangle value: {max_val:.2e}")
        return f"upper triangle is zero (max={max_val:.2e})"

    add_result(results, "causal mask still enforced with use_rope=True", test_causal_mask_with_rope)

    # ------------------------------------------------------------------ #
    # 7. RoPE output differs from baseline (rotation has an effect)        #
    # ------------------------------------------------------------------ #
    def test_rope_changes_output() -> str:
        d_model = GENERAL_CONFIG["d_model"]
        sl      = GENERAL_CONFIG["max_seq_len"]
        x       = torch.randn(batch, sl, d_model)

        original = GENERAL_CONFIG.get("use_rope", False)

        GENERAL_CONFIG["use_rope"] = False
        mha_base = MultiHeadAttention().eval()
        GENERAL_CONFIG["use_rope"] = True
        mha_rope = MultiHeadAttention().eval()
        GENERAL_CONFIG["use_rope"] = original

        # copy weights so the only difference is RoPE
        mha_rope.load_state_dict(mha_base.state_dict(), strict=False)

        with torch.no_grad():
            out_base, _ = mha_base(x)
            out_rope, _ = mha_rope(x)

        if torch.allclose(out_base, out_rope, atol=1e-5):
            raise AssertionError("RoPE and base produced identical output — rope may not be applied")
        delta = (out_base - out_rope).abs().max().item()
        return f"outputs differ by up to {delta:.4f}"

    add_result(results, "use_rope=True produces different output than use_rope=False", test_rope_changes_output)

    # ------------------------------------------------------------------ #
    # 8. Sinusoidal PE is skipped when use_rope=True                       #
    # ------------------------------------------------------------------ #
    def test_sinusoidal_skipped() -> str:
        from text_processing.embedding_classes import PositionalEncoding
        d_model = 64
        sl      = 16
        pe = PositionalEncoding(d_model=d_model, seq_len=sl, dropout=0.0).eval()
        x  = torch.randn(1, sl, d_model)

        original = GENERAL_CONFIG.get("use_rope", False)

        GENERAL_CONFIG["use_rope"] = False
        with torch.no_grad():
            out_base = pe(x)

        GENERAL_CONFIG["use_rope"] = True
        with torch.no_grad():
            out_rope = pe(x)

        GENERAL_CONFIG["use_rope"] = original

        # with use_rope=True, PE should be identity (no signal added), so output == input
        if not torch.allclose(out_rope, x, atol=1e-6):
            raise AssertionError("PE was not skipped — output differs from input when use_rope=True")
        # and base should differ from input (PE was added)
        if torch.allclose(out_base, x, atol=1e-6):
            raise AssertionError("PE had no effect in base mode — sinusoidal signal may be zero")
        return "PE skipped when use_rope=True, applied when use_rope=False"

    add_result(results, "sinusoidal PE is skipped when use_rope=True", test_sinusoidal_skipped)

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
