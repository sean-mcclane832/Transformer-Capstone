import math

import torch
import torch.nn as nn

from utils.config import GENERAL_CONFIG


class InputEmbeddings(nn.Module):
    def __init__(
        self,
        d_model: int = GENERAL_CONFIG["d_model"],
        vocab_size: int = GENERAL_CONFIG["vocab_size"],
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.token_embeddings(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        d_model: int = GENERAL_CONFIG["d_model"],
        seq_len: int = GENERAL_CONFIG["max_seq_len"],
        dropout: float = GENERAL_CONFIG["dropout"],
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.seq_len:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds positional encoding length {self.seq_len}"
            )

        x = x + self.pe[:, :seq_len].requires_grad_(False)
        return self.dropout(x)
