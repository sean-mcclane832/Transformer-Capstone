import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import GENERAL_CONFIG


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Two variants controlled by GENERAL_CONFIG["use_swiglu"]:

    GELU (GPT-2 default, use_swiglu=False):
        x -> W_1 -> GELU -> Dropout -> W_2

    SwiGLU (LLaMA style, use_swiglu=True):
        x -> SiLU(W_1 · x) ⊙ (Wg · x) -> Dropout -> W_2
        d_ff is scaled to 2/3 of the GELU baseline so both variants
        have the same parameter count (SwiGLU uses 3 matrices vs 2).
    """

    def __init__(self):
        super().__init__()

        self.d_model    = GENERAL_CONFIG["d_model"]
        self.dropout_p  = GENERAL_CONFIG["dropout"]
        self.use_swiglu = GENERAL_CONFIG.get("use_swiglu", False)

        if self.use_swiglu:
            # Scale inner dim to 2/3 of baseline so param count matches GELU.
            # GELU: 2 * d_model * d_ff params; SwiGLU: 3 * d_model * d_ff_swi
            # Setting d_ff_swi = 2/3 * d_ff makes both equal.
            d_ff = int(2 / 3 * GENERAL_CONFIG["d_ff"])
            self.W_gate = nn.Linear(self.d_model, d_ff, bias=True)
            self.W_1    = nn.Linear(self.d_model, d_ff, bias=True)
            self.W_2    = nn.Linear(d_ff, self.d_model, bias=True)
        else:
            d_ff = GENERAL_CONFIG["d_ff"]
            self.W_1 = nn.Linear(self.d_model, d_ff, bias=True)
            self.W_2 = nn.Linear(d_ff, self.d_model, bias=True)

        self.W_2.is_residual_projection = True
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x):
        if self.use_swiglu:
            # SwiGLU: gate the hidden state with a sigmoid-linear unit
            x = F.silu(self.W_1(x)) * self.W_gate(x)
            x = self.dropout(x)
            x = self.W_2(x)
        else:
            x = F.gelu(self.W_1(x))
            x = self.dropout(x)
            x = self.W_2(x)
        return x
