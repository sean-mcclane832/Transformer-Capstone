from .token_class import ByteBPETokenizer
from .embedding_classes import InputEmbeddings, PositionalEncoding
import torch
import torch.nn as nn

class TextEmbedder:
    def __init__(self, tokenizer_path: str, d_model: int, vocab_size: int, max_seq_len: int, dropout: float = 0.1):
        self.tokenizer = ByteBPETokenizer.load(tokenizer_path)
        self.input_embeddings = InputEmbeddings(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

    def embed_text(self, text: str) -> torch.Tensor:
        #tokenize
        if text is None:
            ids = self.tokeknizer.encode("", add_bos = False, add_eos = False)
        else:
            ids = self.tokenizer.encode(text, add_bos = True, add_eos = True)
        #add positional encodings and convert to tensor
        ids_tensor = torch.tensor(ids).unsqueeze(0)  #add batch dimension
        embeddings = self.input_embeddings(ids_tensor)
        embeddings = self.positional_encoding(embeddings)
        return embeddings
