import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from utils.config import GENERAL_CONFIG


class ScaledDotAttention(nn.Module):
    def __init__(self, d_model: int = GENERAL_CONFIG["d_model"], dropout: float = GENERAL_CONFIG["dropout"]) -> None:
        super().__init__()
        self.scale = math.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        # Q, K, V: (batch, seq_len, d_model)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, seq_len, seq_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)
        weights = self.dropout(weights)

        output = torch.matmul(weights, V)  # (batch, seq_len, d_model)
        return output, weights

