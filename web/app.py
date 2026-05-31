import sys
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from flask import Flask, Response, render_template, request

from model.gpt import GPT
from model.generate import generate_stream
from text_processing.token_class import ByteBPETokenizer
from utils.config import GENERAL_CONFIG, TOKENIZER_CONFIG

app = Flask(__name__)

# Loaded once at startup, reused across all requests
_model: GPT = None
_tokenizer: ByteBPETokenizer = None
_device: torch.device = None


def _load_model(checkpoint_path: Path, device: torch.device) -> GPT:
    m = GPT().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Support both a raw state_dict and a wrapped training checkpoint
    state = ckpt.get("model", ckpt)
    m.load_state_dict(state)
    m.eval()
    return m


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_route():
    body           = request.get_json(force=True)
    prompt         = body.get("prompt", "")
    max_new_tokens = int(body.get("max_new_tokens", 200))
    temperature    = float(body.get("temperature", 0.8))
    top_k          = int(body.get("top_k", 40))
    top_p          = float(body.get("top_p", 0.9))

    # Frontend sends 0 for "disabled"; convert to None for the sampler
    top_k_arg = top_k if top_k > 0 else None
    top_p_arg = top_p if top_p < 1.0 else None

    def stream():
        ids = _tokenizer.encode(prompt, add_bos=True, add_eos=False)
        idx = torch.tensor([ids], dtype=torch.long, device=_device)

        with torch.no_grad():
            for token_id in generate_stream(
                _model, idx, max_new_tokens,
                temperature=temperature,
                top_k=top_k_arg,
                top_p=top_p_arg,
            ):
                token_text = _tokenizer.decode([token_id])
                yield f"data: {json.dumps(token_text)}\n\n"

        yield 'data: "[DONE]"\n\n'

    return Response(
        stream(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ADRIA streaming web interface")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best.pt",
                        help="Path to model checkpoint (default: checkpoints/best.pt)")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda or cpu (default: cuda if available, else cpu)")
    args = parser.parse_args()

    dev_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _device = torch.device(dev_str)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = ROOT / ckpt_path

    print(f"Loading tokenizer from {TOKENIZER_CONFIG['output']} ...")
    _tokenizer = ByteBPETokenizer()
    _tokenizer.load(TOKENIZER_CONFIG["output"])
    print(f"Tokenizer loaded. Vocab size: {_tokenizer.vocab_size}")

    print(f"Loading model from {ckpt_path} on {_device} ...")
    _model = _load_model(ckpt_path, _device)
    param_count = sum(p.numel() for p in _model.parameters()) / 1e6
    print(f"Model loaded. {param_count:.1f}M parameters.")

    print(f"\nStarting server — open http://localhost:{args.port} in your browser\n")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=False)
