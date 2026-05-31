def checkpoint_name(tier: str, step: int, val_loss: float, arch: str = "base") -> str:
    """
    Build a checkpoint filename encoding size tier, architecture variant, step, and val loss.

    Args:
        tier:     size preset name, e.g. "nano", "small" — matches a key in MODEL_TIERS
        step:     training step number
        val_loss: validation loss at this checkpoint
        arch:     architecture variant tag, e.g. "base", "rope", "gqa", "rope-gqa"
                  defaults to "base" for the sinusoidal / no-frills architecture

    Returns:
        e.g. "adria-small-base-step005000-2.41.pt"

    Examples:
        checkpoint_name("small", 5000, 2.4138)            → "adria-small-base-step005000-2.41.pt"
        checkpoint_name("small", 5000, 2.35, arch="rope") → "adria-small-rope-step005000-2.35.pt"
    """
    return f"adria-{tier}-{arch}-step{step:06d}-{val_loss:.2f}.pt"
