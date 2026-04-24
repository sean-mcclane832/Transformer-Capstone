import torch

from utils.config import GENERAL_CONFIG


def causal_mask(seq_len: int = GENERAL_CONFIG["max_seq_len"], device=None):
    # Lower triangular matrix — device must match the tensor it will be applied to
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask  # (seq_len, seq_len)
