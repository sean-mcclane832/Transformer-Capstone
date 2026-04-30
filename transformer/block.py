import torch.nn as nn

from attention.multi_head import MultiHeadAttention
from transformer.feed_forward import FeedForward
from transformer.layer_norm import LayerNorm
from utils.config import GENERAL_CONFIG


class TransformerBlock(nn.Module):
    """
    Single decoder-only transformer block (Pre-LN).

        x = x + Attention( LayerNorm(x) )
        x = x + FFN(       LayerNorm(x) )
    """

    def __init__(self):
        super().__init__()

        d_model   = GENERAL_CONFIG["d_model"]
        dropout_p = GENERAL_CONFIG["dropout"]

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.attention = MultiHeadAttention()
        self.ffn       = FeedForward()
        self.dropout   = nn.Dropout(dropout_p)

    def forward(self, x, mask=None):
        attn_result = self.attention(self.norm1(x), mask)

        # MultiHeadAttention returns (output, weights) or just output
        if self.attention.return_attn_weights:
            attn_out, weights = attn_result
        else:
            attn_out = attn_result

        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.ffn(self.norm2(x)))

        if self.attention.return_attn_weights:
            return x, weights
        return x
