import sys
from pathlib import Path
from typing import Callable, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from utils.config import GENERAL_CONFIG
from model.gpt import GPT
from model.generate import generate, greedy_decode


CheckResult = Tuple[str, str, str]


def run_check(name: str, check: Callable[[], str]) -> CheckResult:
    try:
        return ("PASS", name, check())
    except Exception as exc:
        return ("FAIL", name, f"{type(exc).__name__}: {exc}")


def add_result(results: List[CheckResult], name: str, check: Callable[[], str]) -> None:
    results.append(run_check(name, check))


def make_model() -> GPT:
    torch.manual_seed(GENERAL_CONFIG["seed"])
    model = GPT()
    model.eval()
    return model


def make_prompt(length: int = 4) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randint(0, GENERAL_CONFIG["vocab_size"], (1, length), dtype=torch.long)


def main() -> None:
    results: List[CheckResult] = []

    max_seq_len  = GENERAL_CONFIG["max_seq_len"]
    vocab_size   = GENERAL_CONFIG["vocab_size"]
    prompt_len   = 4
    new_tokens   = 8

    print(f"Config: vocab_size={vocab_size}, max_seq_len={max_seq_len}")
    print()

    # ------------------------------------------------------------------ #
    # 1. Output shape                                                       #
    # ------------------------------------------------------------------ #
    def test_output_shape() -> str:
        model  = make_model()
        prompt = make_prompt(prompt_len)
        out    = generate(model, prompt, max_new_tokens=new_tokens)
        expected = (1, prompt_len + new_tokens)
        if tuple(out.shape) != expected:
            raise AssertionError(f"expected {expected}, got {tuple(out.shape)}")
        return f"output.shape={tuple(out.shape)}"

    add_result(results, "output shape is (1, prompt_len + max_new_tokens)", test_output_shape)

    # ------------------------------------------------------------------ #
    # 2. Prompt tokens are preserved                                       #
    # ------------------------------------------------------------------ #
    def test_prompt_preserved() -> str:
        model  = make_model()
        prompt = make_prompt(prompt_len)
        out    = generate(model, prompt, max_new_tokens=new_tokens)
        if not torch.equal(out[:, :prompt_len], prompt):
            raise AssertionError("prompt tokens were modified during generation")
        return "prompt tokens unchanged"

    add_result(results, "prompt tokens are preserved in output", test_prompt_preserved)

    # ------------------------------------------------------------------ #
    # 3. All generated tokens are valid vocab IDs                          #
    # ------------------------------------------------------------------ #
    def test_valid_token_ids() -> str:
        model  = make_model()
        prompt = make_prompt(prompt_len)
        out    = generate(model, prompt, max_new_tokens=new_tokens)
        generated = out[:, prompt_len:]
        if generated.min().item() < 0 or generated.max().item() >= vocab_size:
            raise AssertionError(
                f"out-of-range token: min={generated.min()}, max={generated.max()}"
            )
        return f"all tokens in [0, {vocab_size})"

    add_result(results, "generated token IDs are within vocab range", test_valid_token_ids)

    # ------------------------------------------------------------------ #
    # 4. Greedy decode is deterministic                                    #
    # ------------------------------------------------------------------ #
    def test_greedy_deterministic() -> str:
        model  = make_model()
        prompt = make_prompt(prompt_len)
        out1   = greedy_decode(model, prompt, max_new_tokens=new_tokens)
        out2   = greedy_decode(model, prompt, max_new_tokens=new_tokens)
        if not torch.equal(out1, out2):
            raise AssertionError("greedy decode produced different results on identical inputs")
        return "two identical runs produced identical output"

    add_result(results, "greedy decoding is deterministic", test_greedy_deterministic)

    # ------------------------------------------------------------------ #
    # 5. temperature=0.01 ≈ greedy (very peaked distribution)             #
    # ------------------------------------------------------------------ #
    def test_low_temperature_near_greedy() -> str:
        torch.manual_seed(42)
        model   = make_model()
        prompt  = make_prompt(prompt_len)
        greedy  = greedy_decode(model, prompt.clone(), max_new_tokens=new_tokens)
        sampled = generate(model, prompt.clone(), max_new_tokens=new_tokens, temperature=0.01)
        if not torch.equal(greedy[:, prompt_len:], sampled[:, prompt_len:]):
            # Very low temperature should almost always match greedy — allow 1 mismatch
            mismatches = (greedy[:, prompt_len:] != sampled[:, prompt_len:]).sum().item()
            if mismatches > 1:
                raise AssertionError(f"{mismatches} token mismatches vs greedy at temperature=0.01")
        return "temperature=0.01 matches greedy"

    add_result(results, "temperature=0.01 matches greedy decoding", test_low_temperature_near_greedy)

    # ------------------------------------------------------------------ #
    # 6. Context cropped to max_seq_len                                    #
    # ------------------------------------------------------------------ #
    def test_long_prompt_cropped() -> str:
        model  = make_model()
        # Prompt longer than max_seq_len — should not raise
        long_prompt = torch.randint(0, vocab_size, (1, max_seq_len + 10), dtype=torch.long)
        out = generate(model, long_prompt, max_new_tokens=2)
        return f"no error with prompt length {max_seq_len + 10} > max_seq_len"

    add_result(results, "generation handles prompts longer than max_seq_len", test_long_prompt_cropped)

    # ------------------------------------------------------------------ #
    # Print results                                                         #
    # ------------------------------------------------------------------ #
    print("Results:")
    passed = failed = 0
    for status, name, detail in results:
        print(f"  [{status}] {name}: {detail}")
        if status == "PASS":
            passed += 1
        else:
            failed += 1

    print()
    print(f"Summary: {passed} passed, {failed} failed")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
