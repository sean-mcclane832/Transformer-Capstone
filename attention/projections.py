import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import text_processing.text_processor
import utils.config
import utils.config

class AttentionProjections(nn.Module):
    def __init__(self, d_model):
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
