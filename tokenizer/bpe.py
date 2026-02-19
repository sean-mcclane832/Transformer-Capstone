text = 'Hello, world!'
tokens = text.encode('utf-8')
tokens = list(map(int, tokens))
print(tokens)


def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]): # Pythonic way to iterate consecutive elements
        counts[pair] = counts.get(pair, 0) + 1
    return counts