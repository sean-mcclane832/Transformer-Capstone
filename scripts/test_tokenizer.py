import argparse
import sys
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from text_processing.token_class import ByteBPETokenizer
from utils.config import GENERAL_CONFIG, SCRIPT_CONFIG, TOKENIZER_CONFIG


CheckResult = Tuple[str, str, str]


def run_check(name: str, check: Callable[[], str]) -> CheckResult:
    try:
        return ("PASS", name, check())
    except Exception as exc:
        return ("FAIL", name, f"{type(exc).__name__}: {exc}")


def add_result(results: List[CheckResult], name: str, check: Callable[[], str]) -> None:
    results.append(run_check(name, check))


def add_skip(results: List[CheckResult], name: str, reason: str) -> None:
    results.append(("SKIP", name, reason))


def main() -> None:
    parser = argparse.ArgumentParser(description="Test a trained tokenizer on sample text.")
    parser.add_argument("--tokenizer", default=TOKENIZER_CONFIG["output"])
    args = parser.parse_args()

    results: List[CheckResult] = []
    samples = SCRIPT_CONFIG["test_tokenizer"]["samples"]

    print(f"Tokenizer: {args.tokenizer}")
    print()

    tok = None

    # ------------------------------------------------------------------ #
    # 1. Load                                                              #
    # ------------------------------------------------------------------ #
    def test_load() -> str:
        nonlocal tok
        tok = ByteBPETokenizer.load(args.tokenizer)
        return (
            f"vocab_size={tok.vocab_size}, "
            f"bos_id={tok.bos_id}, eos_id={tok.eos_id}"
        )

    load_result = run_check("load tokenizer", test_load)
    results.append(load_result)

    if load_result[0] != "PASS" or tok is None:
        for name in [
            "vocab_size matches config",
            "encode returns IDs within vocab range",
            "BOS token is first when add_bos=True",
            "EOS token is last when add_eos=True",
            "round-trip encode → decode is lossless",
            "encoding without special tokens omits BOS/EOS",
        ]:
            add_skip(results, name, "skipped because tokenizer failed to load")
        _print_results(results)
        raise SystemExit(1)

    # ------------------------------------------------------------------ #
    # 2. vocab_size matches config                                         #
    # ------------------------------------------------------------------ #
    def test_vocab_size() -> str:
        expected = GENERAL_CONFIG["vocab_size"]
        if tok.vocab_size != expected:
            raise AssertionError(
                f"tokenizer vocab_size={tok.vocab_size}, config expects {expected}"
            )
        return f"vocab_size={tok.vocab_size}"

    add_result(results, "vocab_size matches config", test_vocab_size)

    # ------------------------------------------------------------------ #
    # 3. All encoded IDs fall within [0, vocab_size)                      #
    # ------------------------------------------------------------------ #
    def test_ids_in_range() -> str:
        bad = []
        for text in samples:
            ids = tok.encode(text, add_bos=True, add_eos=True)
            out_of_range = [i for i in ids if not (0 <= i < tok.vocab_size)]
            if out_of_range:
                bad.append((text[:20], out_of_range))
        if bad:
            raise AssertionError(f"out-of-range IDs found: {bad}")
        total = sum(len(tok.encode(t, add_bos=True, add_eos=True)) for t in samples)
        return f"all {total} IDs across {len(samples)} samples within [0, {tok.vocab_size})"

    add_result(results, "encode returns IDs within vocab range", test_ids_in_range)

    # ------------------------------------------------------------------ #
    # 4. BOS token is the first ID when add_bos=True                     #
    # ------------------------------------------------------------------ #
    def test_bos_placement() -> str:
        if tok.bos_id is None:
            raise AssertionError("tokenizer has no bos_id")
        for text in samples:
            ids = tok.encode(text, add_bos=True, add_eos=False)
            if ids[0] != tok.bos_id:
                raise AssertionError(
                    f"first token is {ids[0]}, expected bos_id={tok.bos_id} for {ascii(text)}"
                )
        return f"bos_id={tok.bos_id} is first token in all {len(samples)} samples"

    add_result(results, "BOS token is first when add_bos=True", test_bos_placement)

    # ------------------------------------------------------------------ #
    # 5. EOS token is the last ID when add_eos=True                      #
    # ------------------------------------------------------------------ #
    def test_eos_placement() -> str:
        if tok.eos_id is None:
            raise AssertionError("tokenizer has no eos_id")
        for text in samples:
            ids = tok.encode(text, add_bos=False, add_eos=True)
            if ids[-1] != tok.eos_id:
                raise AssertionError(
                    f"last token is {ids[-1]}, expected eos_id={tok.eos_id} for {ascii(text)}"
                )
        return f"eos_id={tok.eos_id} is last token in all {len(samples)} samples"

    add_result(results, "EOS token is last when add_eos=True", test_eos_placement)

    # ------------------------------------------------------------------ #
    # 6. Round-trip: decode(encode(text)) == text                         #
    # ------------------------------------------------------------------ #
    def test_round_trip() -> str:
        failures = []
        for text in samples:
            ids = tok.encode(text, add_bos=True, add_eos=True)
            decoded = tok.decode(ids)
            if decoded != text:
                failures.append(
                    f"  input   : {ascii(text)}\n"
                    f"  decoded : {ascii(decoded)}"
                )
        if failures:
            raise AssertionError(
                f"{len(failures)}/{len(samples)} samples failed round-trip:\n"
                + "\n".join(failures)
            )
        return f"all {len(samples)} samples round-trip losslessly"

    add_result(results, "round-trip encode => decode is lossless", test_round_trip)

    # ------------------------------------------------------------------ #
    # 7. Encoding without special tokens omits BOS/EOS                   #
    # ------------------------------------------------------------------ #
    def test_no_special_tokens() -> str:
        for text in samples:
            ids = tok.encode(text, add_bos=False, add_eos=False)
            if tok.bos_id is not None and ids[0] == tok.bos_id:
                raise AssertionError(
                    f"bos_id={tok.bos_id} present at position 0 when add_bos=False"
                )
            if tok.eos_id is not None and ids[-1] == tok.eos_id:
                raise AssertionError(
                    f"eos_id={tok.eos_id} present at last position when add_eos=False"
                )
        return f"no BOS/EOS in {len(samples)} samples encoded with add_bos=False, add_eos=False"

    add_result(results, "encoding without special tokens omits BOS/EOS", test_no_special_tokens)

    _print_results(results)


def _print_results(results: List[CheckResult]) -> None:
    print("Results:")
    passed = failed = skipped = 0
    for status, name, detail in results:
        print(f"  [{status}] {name}: {detail}")
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
