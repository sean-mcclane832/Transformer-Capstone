import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.token_embeddings = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.token_embeddings(x) * (self.d_model ** 0.5)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #build the positional encoding matrix with shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        #track token positions as a (seq_len, 1) column vector
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)) / d_model))
