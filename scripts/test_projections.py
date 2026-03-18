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
import attention.projections

encoder = text_processing.text_processor.TextEmbedder(
    tokenizer_path=utils.config.TOKENIZER_CONFIG['output'],
    d_model=64,
    vocab_size=utils.config.GENERAL_CONFIG['vocab_size'],
    max_seq_len=64,
    dropout=0.1
)
proj = attention.projections.AttentionProjections(64)

x = encoder.embed_text("hello world")
Q, K, V = proj(x)

print(Q)
print("K shape:", K.shape)
print("V shape:", V.shape)