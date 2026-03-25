#!/usr/bin/env python3
# main.py — Entry point for the LLM Guardrails Firewall
#
# Usage:
#   python -m app.main                  # interactive mode
#   echo "some prompt" | python -m app.main   # pipe mode

import sys
from app.classifier import classify
from app.threshold  import apply_threshold
from app.utils      import log_result, format_result
from app.config     import THRESHOLD, VERBOSE

BANNER = """
╔══════════════════════════════════════════╗
║      LLM Guardrails Firewall  v1.0       ║
║  Protecting your AI pipeline from abuse  ║
╚══════════════════════════════════════════╝
"""


def process_prompt(prompt: str, threshold: float = THRESHOLD) -> dict:
    """
    Full pipeline: classify → threshold → log → return result dict.
    """
    result   = classify(prompt)
    decision = apply_threshold(result["label"], result["confidence"], threshold)

    log_result(
        prompt     = prompt,
        label      = result["label"],
        category   = result["category"],
        confidence = result["confidence"],
        decision   = decision,
    )

    result["decision"] = decision
    return result


def run_interactive():
    print(BANNER)
    print(f"Threshold set to {THRESHOLD:.0%}  (edit app/config.py to change)\n")
    print("Type a prompt and press Enter. Type 'quit' to exit.\n")

    while True:
        try:
            prompt = input(">> Prompt: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not prompt:
            continue
        if prompt.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        result = process_prompt(prompt)
        if VERBOSE:
            print(format_result(
                result["label"],
                result["category"],
                result["confidence"],
                result["decision"],
            ))


def run_pipe():
    """Process prompts from stdin (one per line)."""
    for line in sys.stdin:
        prompt = line.strip()
        if not prompt:
            continue
        result = process_prompt(prompt)
        print(
            f"{result['decision']} | {result['label']} | "
            f"{result['category']} | {result['confidence']:.2%} | "
            f"{prompt[:80]}"
        )


if __name__ == "__main__":
    if sys.stdin.isatty():
        run_interactive()
    else:
        run_pipe()
