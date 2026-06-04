import torch
import torch.nn as nn

from attention.projections import AttentionProjections
from attention.scaled_dot import scaled_dot_product_attention
from attention.mask import causal_mask
from attention.rope import RotaryEmbedding, apply_rotary
from utils.config import GENERAL_CONFIG

class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.d_model   = GENERAL_CONFIG["d_model"]
        self.n_heads   = GENERAL_CONFIG["n_heads"]
        self.dropout_p = GENERAL_CONFIG["dropout"]
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"

        self.d_k = self.d_model // self.n_heads
        self.return_attn_weights = GENERAL_CONFIG["return_attn_weights"]
        self.projections = AttentionProjections(self.d_model)
        self.dropout = nn.Dropout(self.dropout_p)

        self.W_o = nn.Linear(self.d_model, self.d_model, bias=False)
        self.W_o.is_residual_projection = True

        self.use_rope = GENERAL_CONFIG.get("use_rope", False)
        if self.use_rope:
            self.rope = RotaryEmbedding(self.d_k, GENERAL_CONFIG["max_seq_len"])

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.size()
        Q, K, V = self.projections(x)

        # (batch, n_heads, seq_len, d_k)
        Q = Q.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        if self.use_rope:
            cos, sin = self.rope(seq_len)
            Q = apply_rotary(Q, cos, sin)
            K = apply_rotary(K, cos, sin)

        if mask is None:
            mask = causal_mask(seq_len).unsqueeze(0).unsqueeze(0).to(x.device)
        attn_output, weights = scaled_dot_product_attention(Q, K, V, mask, self.dropout if self.training else None)

        # attn_output shape: (batch, n_heads, seq_len, d_k)
        attn_output = attn_output.transpose(1, 2) # → (batch, seq_len, n_heads, d_k)
        attn_output = attn_output.contiguous().view(batch, seq_len, self.d_model)  # → (batch, seq_len, d_model)

        output = self.W_o(attn_output)
        if self.return_attn_weights:
            return output, weights
        return output

