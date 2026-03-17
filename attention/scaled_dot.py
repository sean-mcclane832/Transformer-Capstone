import torch
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Q, K, V: (batch, seq_len, d_model)

    d_k = Q.size(-1)

    # (batch, seq_len, seq_len)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, V)

    return output, attn_weights