from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


class UTF8BPE:
    """UTF-8 byte-level BPE tokenizer with configurable vocabulary size."""

    def __init__(self, vocab_size: int = 32000, special_tokens: Iterable[str] | None = None):
        default_special = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>"]
        self.special_tokens = list(default_special if special_tokens is None else special_tokens)
        min_vocab = len(self.special_tokens) + 256
        if vocab_size < min_vocab:
            raise ValueError(f"vocab_size must be at least {min_vocab} to include UTF-8 byte base vocab + special tokens")
        self.vocab_size = vocab_size
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3
        self.merges: Dict[Tuple[int, int], int] = {}

        self.id_to_bytes: Dict[int, bytes] = {i + len(self.special_tokens): bytes([i]) for i in range(256)}
        for idx, token in enumerate(self.special_tokens):
            self.id_to_bytes[idx] = token.encode("utf-8")
        self.byte_to_id = {v: k for k, v in self.id_to_bytes.items() if k >= len(self.special_tokens)}

    @property
    def base_offset(self) -> int:
        return len(self.special_tokens)

    @property
    def bpe_start_id(self) -> int:
        return self.base_offset + 256

    def _split_chunks(self, text: str) -> List[str]:
        return re.findall(r"\S+|\s+", text)

    def _chunk_to_ids(self, chunk: str) -> List[int]:
        return [self.byte_to_id[bytes([b])] for b in chunk.encode("utf-8")]

    def train(self, text: str, verbose: bool = True) -> None:
        chunks = self._split_chunks(text)
        sequences = [self._chunk_to_ids(c) for c in chunks if c]
        next_id = self.bpe_start_id

        while next_id < self.vocab_size:
            counts = Counter()
            for seq in sequences:
                counts.update(zip(seq, seq[1:]))
            if not counts:
                break

            pair, freq = counts.most_common(1)[0]
            if freq < 2:
                break

            self.merges[pair] = next_id
            self.id_to_bytes[next_id] = self.id_to_bytes[pair[0]] + self.id_to_bytes[pair[1]]

            new_sequences = []
            for seq in sequences:
                merged = []
                i = 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i + 1]) == pair:
                        merged.append(next_id)
                        i += 2
                    else:
                        merged.append(seq[i])
                        i += 1
                new_sequences.append(merged)
            sequences = new_sequences

            if verbose and (next_id - self.bpe_start_id + 1) % 1000 == 0:
                print(f"learned merges: {next_id - self.bpe_start_id + 1}")
            next_id += 1

    def _apply_merges(self, ids: List[int]) -> List[int]:
        if not self.merges:
            return ids
        while True:
            changed = False
            out = []
            i = 0
            while i < len(ids):
                if i < len(ids) - 1 and (ids[i], ids[i + 1]) in self.merges:
                    out.append(self.merges[(ids[i], ids[i + 1])])
                    i += 2
                    changed = True
                else:
                    out.append(ids[i])
                    i += 1
            ids = out
            if not changed:
                return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for chunk in self._split_chunks(text):
            ids.extend(self._apply_merges(self._chunk_to_ids(chunk)))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        parts = []
        for tid in token_ids:
            if skip_special_tokens and tid < self.base_offset:
                continue
            parts.append(self.id_to_bytes.get(tid, self.special_tokens[self.unk_id].encode("utf-8")))
        return b"".join(parts).decode("utf-8", errors="ignore")

    def save(self, path: str | Path) -> None:
        payload = {
            "vocab_size": self.vocab_size,
            "special_tokens": self.special_tokens,
            "merges": [[a, b, c] for (a, b), c in self.merges.items()],
            "id_to_bytes": {str(k): list(v) for k, v in self.id_to_bytes.items()},
        }
        Path(path).write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "UTF8BPE":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(vocab_size=payload["vocab_size"], special_tokens=payload["special_tokens"])
        tok.merges = {(a, b): c for a, b, c in payload["merges"]}
        tok.id_to_bytes = {int(k): bytes(v) for k, v in payload["id_to_bytes"].items()}
        tok.byte_to_id = {v: k for k, v in tok.id_to_bytes.items() if k >= tok.base_offset and len(v) == 1}
        return tok
