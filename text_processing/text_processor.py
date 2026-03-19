import torch

from utils.config import GENERAL_CONFIG, TOKENIZER_CONFIG

from .embedding_classes import InputEmbeddings, PositionalEncoding
from .token_class import ByteBPETokenizer


class TextEmbedder:
    def __init__(
        self,
        tokenizer_path: str = TOKENIZER_CONFIG["output"],
        d_model: int = GENERAL_CONFIG["d_model"],
        vocab_size: int = GENERAL_CONFIG["vocab_size"],
        max_seq_len: int = GENERAL_CONFIG["max_seq_len"],
        dropout: float = GENERAL_CONFIG["dropout"],
    ):
        self.tokenizer = ByteBPETokenizer.load(tokenizer_path)
        self.input_embeddings = InputEmbeddings(d_model, vocab_size)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)

    def embed_text(self, text: str) -> torch.Tensor:
        #tokenize
        if text is None:
            ids = self.tokenizer.encode("", add_bos=False, add_eos=False)
        else:
            ids = self.tokenizer.encode(text, add_bos=True, add_eos=True)
        #add positional encodings and convert to tensor
        ids_tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  #add batch dimension
        embeddings = self.input_embeddings(ids_tensor)
        embeddings = self.positional_encoding(embeddings)
        return embeddings
