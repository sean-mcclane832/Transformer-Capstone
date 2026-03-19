import torch.nn as nn

from utils.config import GENERAL_CONFIG

class AttentionProjections(nn.Module):
    def __init__(self, d_model: int = GENERAL_CONFIG["d_model"]) -> None:
        super().__init__()

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        return Q, K, V
