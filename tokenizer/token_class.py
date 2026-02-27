# tokenizer/token_class.py

import json
from typing import Dict, List, Tuple, Optional


class ByteBPETokenizer:
    """
    Byte-level BPE with optional <bos>/<eos> special tokens.

    Design:
    - Base vocab: 0..255 are raw bytes
    - Learned merges: 256..(vocab_size-3) if using 2 special tokens
    - Special tokens: bos_id=vocab_size-2, eos_id=vocab_size-1
    """

    def __init__(self, vocab_size: int, add_special_tokens: bool = True):
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")

        self.vocab_size = vocab_size
        self.add_special_tokens = add_special_tokens

        self.bos_id: Optional[int] = None
        self.eos_id: Optional[int] = None
        if add_special_tokens:
            if vocab_size < 258:
                raise ValueError("vocab_size must be >= 258 to reserve <bos> and <eos>")
            self.bos_id = vocab_size - 2
            self.eos_id = vocab_size - 1

        # (a, b) -> new_id
        self.merges: Dict[Tuple[int, int], int] = {}

        # id -> bytes (only for non-special tokens)
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    def _get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        counts: Dict[Tuple[int, int], int] = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_once(self, ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        new_ids: List[int] = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
                new_ids.append(new_id)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text: str, min_frequency: int = 2) -> None:
        """
        Train merges on a single text string.
        For larger corpora, read your file(s) into one string and pass it here.
        """
        ids = list(text.encode("utf-8"))

        if self.add_special_tokens:
            merges_target_vocab = self.vocab_size - 2  # reserve bos/eos
        else:
            merges_target_vocab = self.vocab_size

        num_merges = merges_target_vocab - 256
        if num_merges < 0:
            raise ValueError("vocab_size too small after reserving special tokens")

        self.merges = {}

        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break

            pair = max(stats, key=stats.get)
            if stats[pair] < min_frequency:
                break

            new_id = 256 + i
            ids = self._merge_once(ids, pair, new_id)
            self.merges[pair] = new_id
            print(f'Merge {new_id}: out of {num_merges}')

        self._rebuild_vocab()

    def _rebuild_vocab(self) -> None:
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = list(text.encode("utf-8"))

        while True:
            stats = self._get_stats(ids)

            best_pair = None
            best_idx = None
            for pair in stats.keys():
                idx = self.merges.get(pair)
                if idx is None:
                    continue
                if best_idx is None or idx < best_idx:
                    best_idx = idx
                    best_pair = pair

            if best_pair is None or best_idx is None:
                break

            ids = self._merge_once(ids, best_pair, best_idx)

        if add_bos:
            if self.bos_id is None:
                raise ValueError("Tokenizer was created without special tokens")
            ids = [self.bos_id] + ids

        if add_eos:
            if self.eos_id is None:
                raise ValueError("Tokenizer was created without special tokens")
            ids = ids + [self.eos_id]

        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        chunks: List[bytes] = []
        for idx in ids:
            if skip_special_tokens and (idx == self.bos_id or idx == self.eos_id):
                continue
            if idx not in self.vocab:
                # Unknown id, skip rather than crash
                continue
            chunks.append(self.vocab[idx])

        b = b"".join(chunks)
        return b.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        data = {
            "vocab_size": self.vocab_size,
            "add_special_tokens": self.add_special_tokens,
            "bos_id": self.bos_id,
            "eos_id": self.eos_id,
            "merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str) -> "ByteBPETokenizer":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tok = cls(
            vocab_size=int(data["vocab_size"]),
            add_special_tokens=bool(data.get("add_special_tokens", True)),
        )

        # Trust file values so ids remain stable even if you change constructor defaults later
        tok.bos_id = data.get("bos_id", tok.bos_id)
        tok.eos_id = data.get("eos_id", tok.eos_id)

        tok.merges = {(int(a), int(b)): int(idx) for a, b, idx in data["merges"]}
        tok._rebuild_vocab()
        return tok
