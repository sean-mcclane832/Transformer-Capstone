import math

import torch

from attention.softmax import softmax


def scaled_dot_product_attention(Q, K, V, mask=None, dropout=None):
    # Q, K, V: (batch, n_heads, seq_len, d_k)
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)  # (batch, n_heads, seq_len, seq_len)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = softmax(scores, dim=-1)  # (batch, n_heads, seq_len, seq_len)

    if dropout is not None:
        weights = dropout(weights)

    output = torch.matmul(weights, V)  # (batch, n_heads, seq_len, d_k)
    return output, weights

