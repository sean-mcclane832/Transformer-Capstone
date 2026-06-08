import torch.nn as nn

from attention.multi_head import MultiHeadAttention
from transformer.feed_forward import FeedForward
from transformer.layer_norm import LayerNorm
from utils.config import GENERAL_CONFIG


class TransformerBlock(nn.Module):
    """
    Transformer block with Pre-LN or Post-LN controlled by GENERAL_CONFIG["pre_ln"].

    Pre-LN (GPT-2, default — pre_ln=True):
        x = x + dropout( Attn( LN(x) ) )
        x = x + dropout( FFN(  LN(x) ) )

    Post-LN (original Vaswani et al. — pre_ln=False):
        x = LN( x + dropout( Attn(x) ) )
        x = LN( x + dropout( FFN(x)  ) )

    Pre-LN is more training-stable; Post-LN is included for ablation.
    """

    def __init__(self):
        super().__init__()

        d_model   = GENERAL_CONFIG["d_model"]
        dropout_p = GENERAL_CONFIG["dropout"]
        self.pre_ln = GENERAL_CONFIG.get("pre_ln", True)

        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.attention = MultiHeadAttention()
        self.ffn       = FeedForward()
        self.dropout   = nn.Dropout(dropout_p)

    def forward(self, x, mask=None):
        if self.pre_ln:
            attn_result = self.attention(self.norm1(x), mask)
            if self.attention.return_attn_weights:
                attn_out, weights = attn_result
            else:
                attn_out = attn_result
            x = x + self.dropout(attn_out)
            x = x + self.dropout(self.ffn(self.norm2(x)))
        else:
            # Post-LN: apply norm after the residual add
            attn_result = self.attention(x, mask)
            if self.attention.return_attn_weights:
                attn_out, weights = attn_result
            else:
                attn_out = attn_result
            x = self.norm1(x + self.dropout(attn_out))
            x = self.norm2(x + self.dropout(self.ffn(x)))

        if self.attention.return_attn_weights:
            return x, weights
        return x
