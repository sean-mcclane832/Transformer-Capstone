import torch
import torch.nn as nn


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings — Su et al., 2021 (RoFormer).

    Encodes position by rotating Q and K vectors rather than adding a
    positional bias to token embeddings. The rotation angle at dimension
    index i and sequence position m is m * θ_i where:

        θ_i = 1 / 10000^(2i / d_k)

    Because rotation is applied before the dot product, attention scores
    naturally depend on relative position: Q_m · K_n = f(m − n).

    Usage:
        rope      = RotaryEmbedding(d_k, max_seq_len)
        cos, sin  = rope(seq_len)
        Q_rotated = apply_rotary(Q, cos, sin)
        K_rotated = apply_rotary(K, cos, sin)
    """

    def __init__(self, d_k: int, max_seq_len: int):
        super().__init__()
        # θ_i = 1 / 10000^(2i / d_k) for i in 0 .. d_k // 2
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, d_k, 2).float() / d_k))
        self.register_buffer("inv_freq", inv_freq)

        # precompute cos/sin tables for every position up to max_seq_len
        t     = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)          # (max_seq_len, d_k // 2)
        emb   = torch.cat([freqs, freqs], dim=-1) # (max_seq_len, d_k) — duplicate freqs across both halves
        self.register_buffer("cos_table", emb.cos())
        self.register_buffer("sin_table", emb.sin())

    def forward(self, seq_len: int):
        # returns tables cropped to the current sequence length
        return self.cos_table[:seq_len], self.sin_table[:seq_len]


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x:        (batch, n_heads, seq_len, d_k)
    # cos, sin: (seq_len, d_k) — unsqueeze to broadcast over batch and heads
    cos = cos.unsqueeze(0).unsqueeze(0)  # → (1, 1, seq_len, d_k)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + _rotate_half(x) * sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # splits last dim in half: [x1 | x2] → [-x2 | x1]
    # this implements the pairwise rotation without an explicit loop
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)
