import torch

from utils.config import GENERAL_CONFIG


def causal_mask(seq_len: int = GENERAL_CONFIG["max_seq_len"]):
    # Lower triangular matrix
    mask = torch.tril(torch.ones(seq_len, seq_len))
    return mask  # (seq_len, seq_len)
