# ── DEFERRED — do not implement until the model is trained and generating ──────
#
# TurboQuant KV Cache — post-training inference optimization.
#
# Design: Option B from the paper — implemented inside the attention layer as
# TurboQuantKVCache. Use the `prod` variant (unbiased for inner-product
# estimation, which is exactly what Q · K^T is).
#
# Implement only after:
#   1. Full training run is complete
#   2. Model is generating coherent text via cli.py
#   3. Baseline inference speed is measured
#
# References:
#   - TurboQuant paper (prod variant)
#   - Milestone 19 in CLAUDE.md
# ──────────────────────────────────────────────────────────────────────────────


class TurboQuantKVCache:
    """Placeholder — not yet implemented."""
    pass
