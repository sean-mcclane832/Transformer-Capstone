import torch.nn as nn

from utils.config import GENERAL_CONFIG

class AttentionProjections(nn.Module):
    def __init__(
        self,
        d_model:    int = GENERAL_CONFIG["d_model"],
        n_heads:    int = GENERAL_CONFIG["n_heads"],
        n_kv_heads: int = None,
    ) -> None:
        super().__init__()

        self.n_kv_heads = n_kv_heads or n_heads
        d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)                          # (d_model → n_heads * d_k)
        self.W_k = nn.Linear(d_model, self.n_kv_heads * d_k)            # smaller when n_kv_heads < n_heads
        self.W_v = nn.Linear(d_model, self.n_kv_heads * d_k)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        return Q, K, V
