import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from text_processing.embedding_classes import InputEmbeddings, PositionalEncoding
from transformer.block import TransformerBlock
from transformer.layer_norm import LayerNorm
from utils.config import GENERAL_CONFIG


class GPT(nn.Module):
    """
    ADRIA — Attention-Driven Recursive Inference Architecture.

    Decoder-only transformer trained via next-token prediction. Every component
    (tokenizer, attention, FFN, LayerNorm, training loop) is hand-implemented
    from scratch in PyTorch as a senior capstone project.

    Pipeline:
        token IDs (batch, seq_len)
          -> token embeddings + positional encoding
          -> N transformer blocks (Pre-LN)
          -> final LayerNorm
          -> LM head (linear -> vocab logits)
    """

    def __init__(self):
        super().__init__()

        self.d_model     = GENERAL_CONFIG["d_model"]
        self.vocab_size  = GENERAL_CONFIG["vocab_size"]
        self.max_seq_len = GENERAL_CONFIG["max_seq_len"]
        self.n_layers    = GENERAL_CONFIG["n_layers"]
        self.dropout_p   = GENERAL_CONFIG["dropout"]

        self.token_emb = InputEmbeddings(self.d_model, self.vocab_size)
        self.pos_enc   = PositionalEncoding(self.d_model, self.max_seq_len, self.dropout_p)

        self.blocks = nn.ModuleList([TransformerBlock() for _ in range(self.n_layers)])

        self.ln_f    = LayerNorm(self.d_model)
        self.lm_head = nn.Linear(self.d_model, self.vocab_size, bias=False)

        if GENERAL_CONFIG.get("tie_weights", True):
            self.lm_head.weight = self.token_emb.token_embeddings.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Residual projections (W_o in attention, W_2 in FFN) are scaled down
        # by 1/sqrt(2 * n_layers) to keep the residual stream variance stable
        # as depth increases — each block adds to the stream, so without scaling
        # the variance grows linearly with n_layers.
        # I don't have a great way to identify these projections, so I'm using a custom attribute set in the TransformerBlock definition.
        if hasattr(module, "is_residual_projection"):
            module.weight.data *= (2 * self.n_layers) ** -0.5

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        batch, seq_len = idx.size()
        if seq_len > self.max_seq_len:
            raise ValueError(f"Input sequence length {seq_len} exceeds max_seq_len {self.max_seq_len}")

        x = self.token_emb(idx)
        x = self.pos_enc(x)

        for block in self.blocks:
            block_out = block(x)
            x = block_out[0] if isinstance(block_out, tuple) else block_out

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))

        return logits, loss


# if __name__ == "__main__":
#     m = GPT()
#     idx = torch.randint(0, GENERAL_CONFIG["vocab_size"], (2, 16))

#     logits, loss = m(idx)
#     print(f"logits.shape = {logits.shape}")  # (2, 16, 8192)
#     print(f"loss (no targets) = {loss}")     # None

#     targets = torch.randint(0, GENERAL_CONFIG["vocab_size"], (2, 16))
#     logits, loss = m(idx, targets)
#     print(f"loss (with targets) = {loss.item():.4f}")  # ~9.0 (log vocab_size)
