"""
cli.py — MiniGPT text generation interface

Usage:
    python scripts/cli.py --checkpoint checkpoints/best.pt --prompt "To be or not to be"
    python scripts/cli.py --checkpoint checkpoints/best.pt --prompt "Call me Ishmael" --temperature 0.8 --top-k 40
    python scripts/cli.py --checkpoint checkpoints/best.pt --prompt "The Great" --top-p 0.9 --max-new-tokens 300
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from model.gpt import GPT
from model.generate import generate
from text_processing.token_class import ByteBPETokenizer
from utils.config import GENERAL_CONFIG, TOKENIZER_CONFIG


def load_model(checkpoint_path: str, device: torch.device) -> GPT:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = GPT().to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def load_tokenizer() -> ByteBPETokenizer:
    tokenizer = ByteBPETokenizer()
    tokenizer.load(TOKENIZER_CONFIG["output"])
    return tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text with MiniGPT")
    parser.add_argument("--checkpoint",     type=str,   required=True,
                        help="Path to a .pt checkpoint file")
    parser.add_argument("--prompt",         type=str,   required=True,
                        help="Prompt text to condition generation on")
    parser.add_argument("--max-new-tokens", type=int,   default=200,
                        help="Number of tokens to generate (default: 200)")
    parser.add_argument("--temperature",    type=float, default=1.0,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-k",          type=int,   default=None,
                        help="Top-k filtering (e.g. 40); disabled if not set")
    parser.add_argument("--top-p",          type=float, default=None,
                        help="Nucleus sampling threshold (e.g. 0.9); disabled if not set")
    parser.add_argument("--greedy",         action="store_true",
                        help="Use greedy decoding instead of sampling")
    parser.add_argument("--device",         type=str,   default=GENERAL_CONFIG["device"],
                        help="Device to run on (default: from config)")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading tokenizer from {TOKENIZER_CONFIG['output']} ...")
    tokenizer = load_tokenizer()

    print(f"Loading model from {args.checkpoint} ...")
    model = load_model(args.checkpoint, device)

    # Encode prompt (no BOS/EOS — generation adds its own context)
    prompt_ids = tokenizer.encode(args.prompt, add_bos=False, add_eos=False)
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    print(f"\nPrompt: {args.prompt!r}")
    print(f"Generating {args.max_new_tokens} tokens "
          f"(temperature={args.temperature}, top_k={args.top_k}, top_p={args.top_p}) ...\n")
    print("-" * 60)

    if args.greedy:
        from model.generate import greedy_decode
        output_ids = greedy_decode(model, idx, args.max_new_tokens)
    else:
        output_ids = generate(
            model, idx,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
        )

    # Decode only the newly generated tokens (skip the prompt)
    generated_ids = output_ids[0, len(prompt_ids):].tolist()
    print(args.prompt + tokenizer.decode(generated_ids))
    print("-" * 60)


if __name__ == "__main__":
    main()
