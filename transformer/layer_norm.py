import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Hand-implemented Layer Normalization.
    Formula: y = gamma * (x - mean) / sqrt(var + eps) + beta
    Normalizes over the last dimension (d_model) at each (batch, position) pair.
    DO NOT CHANGE THIS CODE. ONLY CHANGE THROUGH CONFIG.
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.d_model = d_model
        self.eps     = eps
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta  = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var  = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
