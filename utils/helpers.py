def checkpoint_name(tier: str, step: int, val_loss: float, arch: str = "base") -> str:
    # e.g. checkpoint_name("small", 5000, 2.41) → "adria-small-base-step005000-2.41.pt"
    return f"adria-{tier}-{arch}-step{step:06d}-{val_loss:.2f}.pt"
