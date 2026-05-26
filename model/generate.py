import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F
from model.gpt import GPT
from utils.config import GENERAL_CONFIG


def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    top_p: float | None = None,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """
    Autoregressively sample tokens from a prompt.

    Args:
        model:          trained GPT (set to eval mode internally)
        idx:            (1, seq_len) integer prompt token IDs
        max_new_tokens: maximum number of tokens to generate
        temperature:    logit scaling before sampling — < 1 sharpens, > 1 flattens
        top_k:          if set, zero out all logits outside the top-k before sampling
        top_p:          if set, keep the smallest nucleus of tokens whose cumulative
                        probability >= p (nucleus / top-p sampling)
        eos_token_id:   if set, stop early when this token is sampled

    Returns:
        (1, seq_len + n_generated) token IDs including the prompt
    """
    model.eval()
    max_seq_len = GENERAL_CONFIG["max_seq_len"]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop context so positional encoding never exceeds max_seq_len
            idx_cond = idx if idx.size(1) <= max_seq_len else idx[:, -max_seq_len:]

            # Forward — only the last position's logits matter
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]   # (1, vocab_size)

            # 1. Temperature — scale before any filtering
            if temperature != 1.0:
                logits = logits / temperature

            # 2. Top-k — zero out everything outside the top k
            if top_k is not None:
                logits = _apply_top_k(logits, top_k)

            # 3. Top-p (nucleus) — zero out the long tail beyond cumulative prob p
            if top_p is not None:
                logits = _apply_top_p(logits, top_p)

            # 4. Sample from the resulting distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

            idx = torch.cat([idx, next_token], dim=1)

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

    return idx


def _apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    # Keep the top-k logits; set everything else to -inf
    k = min(k, logits.size(-1))
    values, _ = torch.topk(logits, k, dim=-1)
    threshold = values[:, -1, None]    # kth-largest value per row
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    # Sort descending, compute cumulative softmax probs, remove tokens
    # once cumulative probability exceeds p
    sorted_logits, sorted_indices = torch.sort(logits, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remove tokens where cumulative prob already exceeds p (shift right by 1
    # so we always keep at least the top token)
    remove_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits = sorted_logits.masked_fill(remove_mask, float("-inf"))

    # Scatter back to original ordering
    logits = torch.zeros_like(logits).scatter(-1, sorted_indices, sorted_logits)
    return logits


def greedy_decode(model: GPT, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
    """Deterministic greedy decoding — always picks the highest-probability token."""
    model.eval()
    max_seq_len = GENERAL_CONFIG["max_seq_len"]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= max_seq_len else idx[:, -max_seq_len:]
            logits, _ = model(idx_cond)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
            idx = torch.cat([idx, next_token], dim=1)

    return idx
