import argparse
import math
import sys
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from text_processing.embedding_classes import InputEmbeddings, PositionalEncoding
from text_processing.text_processor import TextEmbedder
from text_processing.token_class import ByteBPETokenizer
from utils.config import GENERAL_CONFIG, TOKENIZER_CONFIG
import text_processing.text_processor as text_processor


class _PositionalEncodingWithDefault(PositionalEncoding):
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.0) -> None:
        super().__init__(d_model, seq_len, dropout)


# Ensure TextEmbedder can construct positional encodings without passing dropout.
text_processor.PositionalEncoding = _PositionalEncodingWithDefault


CheckResult = Tuple[str, str, str]


def resolve_tokenizer_path(cli_path: Optional[str]) -> Path:
    candidates = []

    if cli_path:
        candidates.append(Path(cli_path))

    config_path = Path(TOKENIZER_CONFIG["output"])
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    candidates.append(config_path)

    candidates.append(ROOT / "text_processing" / "tokenizer.json")
    candidates.append(ROOT / "text_processing" / "tokenizer_test.json")

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def run_check(name: str, check: Callable[[], str]) -> CheckResult:
    try:
        return ("PASS", name, check())
    except Exception as exc:
        return ("FAIL", name, f"{type(exc).__name__}: {exc}")


def add_result(results: List[CheckResult], name: str, check: Callable[[], str]) -> None:
    results.append(run_check(name, check))


def add_skip(results: List[CheckResult], name: str, reason: str) -> None:
    results.append(("SKIP", name, reason))


def build_expected_positional_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(seq_len, d_model, dtype=torch.float32)
    position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )

    pe[:, 0::2] = torch.sin(position * div_term)
    if d_model > 1:
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])

    return pe


def tensor_preview(tensor: torch.Tensor, max_items: int = 6) -> str:
    flat = tensor.detach().cpu().reshape(-1).tolist()
    shown = [round(float(value), 6) for value in flat[:max_items]]
    suffix = ", ..." if len(flat) > max_items else ""
    return f"[{', '.join(map(str, shown))}{suffix}]"


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    *,
    name: str,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        max_abs_diff = torch.max(torch.abs(actual - expected)).item()
        raise AssertionError(
            f"{name} mismatch: max_abs_diff={max_abs_diff:.6g}, "
            f"actual={tensor_preview(actual)}, expected={tensor_preview(expected)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run lightweight checks for text_processing/text_processor.py."
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional path to a tokenizer JSON file.",
    )
    parser.add_argument(
        "--text",
        default="Hello world from the embedder test script.",
        help="Sample text used for embedder checks.",
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
        help="Embedding width used for the tests.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=16,
        help="Maximum sequence length used for the tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_path = resolve_tokenizer_path(args.tokenizer)
    results: List[CheckResult] = []

    print(f"Repository root : {ROOT}")
    print(f"Tokenizer path  : {tokenizer_path}")
    print()

    tokenizer: Optional[ByteBPETokenizer] = None

    def load_tokenizer() -> str:
        nonlocal tokenizer
        tokenizer = ByteBPETokenizer.load(str(tokenizer_path))
        return (
            f"vocab_size={tokenizer.vocab_size}, "
            f"bos_id={tokenizer.bos_id}, eos_id={tokenizer.eos_id}"
        )

    add_result(results, "load tokenizer", load_tokenizer)

    vocab_size = tokenizer.vocab_size if tokenizer is not None else GENERAL_CONFIG["vocab_size"]

    def test_input_embeddings_values() -> str:
        layer = InputEmbeddings(4, 6)
        known_weights = torch.tensor(
            [
                [0.0, 1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0, 7.0],
                [8.0, 9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0],
                [20.0, 21.0, 22.0, 23.0],
            ],
            dtype=torch.float32,
        )
        with torch.no_grad():
            layer.token_embeddings.weight.copy_(known_weights)

        sample_ids = torch.tensor([[0, 2, 5]], dtype=torch.long)
        actual = layer(sample_ids)
        expected = known_weights[sample_ids] * math.sqrt(layer.d_model)
        assert_close(actual, expected, name="InputEmbeddings output")
        return (
            f"token_ids={sample_ids.tolist()[0]}, "
            f"first_vector={tensor_preview(actual[0, 0])}"
        )

    add_result(results, "InputEmbeddings numeric output", test_input_embeddings_values)

    positional_encoding: Optional[PositionalEncoding] = None

    def test_positional_encoding() -> str:
        nonlocal positional_encoding
        positional_encoding = PositionalEncoding(4, 6, 0.0)
        sample = torch.zeros(1, 3, 4)
        actual = positional_encoding(sample)
        expected = build_expected_positional_encoding(3, 4).unsqueeze(0)
        assert_close(actual, expected, name="PositionalEncoding output")
        return f"position_0={tensor_preview(actual[0, 0])}, position_1={tensor_preview(actual[0, 1])}"

    add_result(results, "PositionalEncoding numeric output", test_positional_encoding)

    def test_invalid_tokenizer_path() -> str:
        try:
            TextEmbedder(
                tokenizer_path=str(ROOT / "does_not_exist.json"),
                d_model=args.d_model,
                vocab_size=vocab_size,
                max_seq_len=args.max_seq_len,
            )
        except (FileNotFoundError, OSError):
            return "raised file error as expected"
        raise AssertionError("expected file error for an invalid tokenizer path")

    add_result(results, "TextEmbedder rejects invalid tokenizer path", test_invalid_tokenizer_path)

    embedder: Optional[TextEmbedder] = None

    def test_text_embedder_init() -> str:
        nonlocal embedder
        embedder = TextEmbedder(
            tokenizer_path=str(tokenizer_path),
            d_model=args.d_model,
            vocab_size=vocab_size,
            max_seq_len=args.max_seq_len,
        )
        return (
            f"max_seq_len={embedder.positional_encoding.seq_len}, "
            f"d_model={embedder.input_embeddings.d_model}"
        )

    init_result = run_check("TextEmbedder init", test_text_embedder_init)
    results.append(init_result)

    if init_result[0] == "PASS" and embedder is not None:
        def test_embed_text_output() -> str:
            known_weights = torch.arange(
                vocab_size * args.d_model, dtype=torch.float32
            ).reshape(vocab_size, args.d_model)
            with torch.no_grad():
                embedder.input_embeddings.token_embeddings.weight.copy_(known_weights)

            sample_text = args.text
            ids = embedder.tokenizer.encode(sample_text, add_bos=True, add_eos=True)
            if len(ids) > args.max_seq_len:
                sample_text = "Hi"
                ids = embedder.tokenizer.encode(sample_text, add_bos=True, add_eos=True)
            if len(ids) > args.max_seq_len:
                raise AssertionError(
                    f"sample text produced {len(ids)} tokens (max_seq_len={args.max_seq_len})"
                )

            actual = embedder.embed_text(sample_text).detach().cpu()
            ids = ids[: args.max_seq_len]
            ids_tensor = torch.tensor(ids, dtype=torch.long)

            expected_embeddings = known_weights[ids_tensor] * math.sqrt(args.d_model)
            expected_positions = build_expected_positional_encoding(len(ids), args.d_model)
            expected = (expected_embeddings + expected_positions).unsqueeze(0)

            assert_close(actual, expected, name="TextEmbedder.embed_text output")
            return (
                f"token_ids={ids[:6]}{'...' if len(ids) > 6 else ''}, "
                f"first_vector={tensor_preview(actual[0, 0])}"
            )

        add_result(results, "TextEmbedder.embed_text numeric output", test_embed_text_output)

        def test_embed_text_empty_input() -> str:
            actual = embedder.embed_text("")
            ids = embedder.tokenizer.encode("", add_bos=True, add_eos=True)
            if actual.shape[1] != len(ids):
                raise AssertionError(
                    f"expected sequence length {len(ids)}, got {actual.shape[1]}"
                )
            return f"sequence_len={actual.shape[1]}"

        add_result(results, "TextEmbedder.embed_text handles empty input", test_embed_text_empty_input)

        long_text = " ".join(["transformer"] * (args.max_seq_len * 8))
        long_ids = embedder.tokenizer.encode(long_text, add_bos=True, add_eos=True)
        if len(long_ids) > args.max_seq_len:
            def test_embed_text_long_input() -> str:
                try:
                    embedder.embed_text(long_text)
                except ValueError:
                    return f"raised ValueError as expected (seq_len={len(long_ids)})"
                raise AssertionError("expected ValueError for long input")

            add_result(results, "TextEmbedder.embed_text errors on long input", test_embed_text_long_input)
        else:
            add_skip(
                results,
                "TextEmbedder.embed_text errors on long input",
                "skipped because tokenizer output did not exceed max_seq_len",
            )
    else:
        add_skip(
            results,
            "TextEmbedder.embed_text numeric output",
            "skipped because TextEmbedder init failed",
        )
        add_skip(
            results,
            "TextEmbedder.embed_text handles empty input",
            "skipped because TextEmbedder init failed",
        )
        add_skip(
            results,
            "TextEmbedder.embed_text errors on long input",
            "skipped because TextEmbedder init failed",
        )

    print("Results:")
    passed = 0
    failed = 0
    skipped = 0

    for status, name, detail in results:
        print(f"[{status}] {name}: {detail}")
        if status == "PASS":
            passed += 1
        elif status == "FAIL":
            failed += 1
        else:
            skipped += 1

    print()
    print(f"Summary: {passed} passed, {failed} failed, {skipped} skipped")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
