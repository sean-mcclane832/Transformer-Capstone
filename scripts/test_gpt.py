import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.gpt import GPT
m = GPT()
print(m.lm_head.weight.data_ptr() == m.token_emb.token_embeddings.weight.data_ptr())
