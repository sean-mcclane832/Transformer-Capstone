
with open('C:\\Users\\SeanM\\OneDrive\\Documents\\GitHub\\Transformer-Capstone\\data\\raw\\input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts



def merge(ids, pair, new_id):
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids

#debug helper for a quick merge check: print(merge(tokens, top_pair, 256))

#long sample text was used here earlier during unicode testing
tokens = text.encode('utf-8')
tokens = list(map(int, tokens))
#debug helper: print(tokens)

vocab_size = 5000
num_merges = vocab_size - 256
ids = list(tokens)

merges = {}
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx

print("tokens length:", len(tokens))
print("ids length:", len(ids))
print(f"compression ratio: {len(tokens) / len(ids):.2f}X")

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
  #turn token ids back into a python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

print(decode([126]))

def encode(text: str):
    ids = list(text.encode("utf-8"))

    while True:
        stats = get_stats(ids)
        best_pair = None
        best_idx = None

        for pair in stats.keys():
            if pair in merges:
                idx = merges[pair]
                if best_idx is None or idx < best_idx:
                    best_idx = idx
                    best_pair = pair

        if best_pair is None:
            break

        ids = merge(ids, best_pair, best_idx)

    return ids

sample = text
enc = encode(sample)
dec = decode(enc)

print("original:", sample)
print("encoded length:", len(enc))
print("decoded:", dec)
print("match:", sample == dec)
 

