from .token_class import ByteBPETokenizer
from .embedding_classes import InputEmbeddings, PositionalEncoding
import torch
import torch.nn as nn
from typing import Optional

class TextEmbedder(nn.Module):
    def __init__(self, tokenizer_path: str, d_model: int, vocab_size: int, max_seq_len: int, dropout: float = 0.1, device: Optional[torch.device] = None):
        super().__init__()
        if device is None:
            device = torch.device('cpu')
        self.device = device
        
        try:
            self.tokenizer = ByteBPETokenizer.load(tokenizer_path)
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer from {tokenizer_path}: {e}")
        
        self.input_embeddings = InputEmbeddings(d_model, vocab_size).to(self.device)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout).to(self.device)

    def embed_text(self, text: str) -> torch.Tensor:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")
        
        #tokenize the input text
        ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        
        #cap the sequence at the model max length
        if len(ids) > self.positional_encoding.seq_len:
            ids = ids[:self.positional_encoding.seq_len]
        
        #turn ids into a tensor and run embedding layers
        ids_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(self.device)  #add batch dimension
        embeddings = self.input_embeddings(ids_tensor)
        embeddings = self.positional_encoding(embeddings)
        return embeddings
