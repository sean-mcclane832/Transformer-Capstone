import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def _merge(ids, pair, new_id):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


try:
    from tokenizer.token_class import ByteBPETokenizer  # type: ignore
except Exception:
    ByteBPETokenizer = None


if ByteBPETokenizer is None:
    class ByteBPETokenizer:
        def __init__(self, vocab_size=4096, add_special_tokens=True):
            self.vocab_size = int(vocab_size)
            self.use_special_tokens = bool(add_special_tokens)
            self.merges = {}
            self.vocab = {i: bytes([i]) for i in range(256)}
            self.special_tokens = {}

        def _rebuild_vocab(self):
            self.vocab = {i: bytes([i]) for i in range(256)}
            for (p0, p1), idx in sorted(self.merges.items(), key=lambda x: x[1]):
                self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

        def train(self, text: str):
            ids = list(text.encode("utf-8"))
            target_merges = max(self.vocab_size - 256, 0)
            self.merges = {}

            for i in range(target_merges):
                stats = _get_stats(ids)
                if not stats:
                    break
                pair = max(stats, key=stats.get)
                new_id = 256 + i
                ids = _merge(ids, pair, new_id)
                self.merges[pair] = new_id

            self._rebuild_vocab()

            if self.use_special_tokens:
                next_id = (max(self.vocab) + 1) if self.vocab else 256
                for name in ("<pad>", "<bos>", "<eos>", "<unk>"):
                    self.special_tokens[name] = next_id
                    next_id += 1

        def encode(self, text: str, add_bos=False, add_eos=False):
            ids = list(text.encode("utf-8"))

            while len(ids) >= 2:
                stats = _get_stats(ids)
                best_pair = None
                best_idx = None
                for pair in stats:
                    idx = self.merges.get(pair)
                    if idx is None:
                        continue
                    if best_idx is None or idx < best_idx:
                        best_idx = idx
                        best_pair = pair
                if best_pair is None:
                    break
                ids = _merge(ids, best_pair, best_idx)

            out = []
            if add_bos and "<bos>" in self.special_tokens:
                out.append(self.special_tokens["<bos>"])
            out.extend(ids)
            if add_eos and "<eos>" in self.special_tokens:
                out.append(self.special_tokens["<eos>"])
            return out

        def decode(self, ids):
            pieces = []
            special_ids = set(self.special_tokens.values())
            for idx in ids:
                if idx in special_ids:
                    continue
                token_bytes = self.vocab.get(idx)
                if token_bytes is not None:
                    pieces.append(token_bytes)
            return b"".join(pieces).decode("utf-8", errors="replace")

        def save(self, path):
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "vocab_size": self.vocab_size,
                "add_special_tokens": self.use_special_tokens,
                "merges": [[a, b, idx] for (a, b), idx in sorted(self.merges.items(), key=lambda x: x[1])],
                "special_tokens": self.special_tokens,
            }
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        @classmethod
        def load(cls, path):
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            tok = cls(
                vocab_size=payload.get("vocab_size", 4096),
                add_special_tokens=payload.get("add_special_tokens", True),
            )
            tok.merges = {
                (int(a), int(b)): int(idx)
                for a, b, idx in payload.get("merges", [])
            }
            tok.special_tokens = {
                str(k): int(v) for k, v in payload.get("special_tokens", {}).items()
            }
            tok._rebuild_vocab()
            return tok


def main():
    input_path = ROOT / "data" / "raw" / "input.txt"
    output_path = ROOT / "tokenizer" / "tokenizer.json"

    text = input_path.read_text(encoding="utf-8")

    tok = ByteBPETokenizer(vocab_size=4096, add_special_tokens=True)
    tok.train(text)
    tok.save(output_path)

    tok2 = ByteBPETokenizer.load(output_path)
    ids = tok2.encode("hello 😄", add_bos=True, add_eos=True)
    print(ids[:20])
    print(tok2.decode(ids))


if __name__ == "__main__":
    main()
