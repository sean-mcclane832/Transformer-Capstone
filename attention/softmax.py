import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # Subtract max for numerical stability — prevents exp() overflow
    # Subtracting a constant doesn't change the output (numerator and denominator scale equally)
    x = x - x.max(dim=dim, keepdim=True).values
    exp_x = torch.exp(x)
    return exp_x / exp_x.sum(dim=dim, keepdim=True)
