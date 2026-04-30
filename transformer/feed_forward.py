import torch.nn as nn

from utils.config import GENERAL_CONFIG


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    The same MLP is applied independently at every sequence position. Attention
    already handled cross-token mixing — this module's job is per-token nonlinear
    processing. The inner dimension expands to d_ff (= 4 * d_model by convention),
    runs through a nonlinearity, then contracts back to d_model.

    Architecture:
        x -> Linear(d_model -> d_ff) -> GELU -> Dropout -> Linear(d_ff -> d_model)
    """

    def __init__(self):
        super().__init__()

        self.d_model   = GENERAL_CONFIG["d_model"]
        self.d_ff      = GENERAL_CONFIG["d_ff"]
        self.dropout_p = GENERAL_CONFIG["dropout"]

        # Expand: d_model -> d_ff
        self.W_1 = nn.Linear(self.d_model, self.d_ff, bias=True)

        # GPT-2 activation
        self.activation = nn.GELU()

        # Dropout on activations
        self.dropout = nn.Dropout(self.dropout_p)

        # Contract: d_ff -> d_model
        self.W_2 = nn.Linear(self.d_ff, self.d_model, bias=True)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        x = self.W_1(x)           # -> (batch, seq_len, d_ff)
        x = self.activation(x)    # -> (batch, seq_len, d_ff)
        x = self.dropout(x)       # -> (batch, seq_len, d_ff)
        x = self.W_2(x)           # -> (batch, seq_len, d_model)
        return x
