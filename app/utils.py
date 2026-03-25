# utils.py — Text cleaning helpers and a simple CSV logger

import re
import csv
import os
from datetime import datetime
from app.config import LOGS_PATH


def clean_text(text: str) -> str:
    """Lowercase and strip extra whitespace from a prompt."""
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)          # collapse multiple spaces
    text = re.sub(r"[^\w\s\.\,\?\!\'\-]", " ", text)  # drop unusual chars
    return text


def log_result(prompt: str, label: str, category: str,
               confidence: float, decision: str) -> None:
    """Append a single classification result to the CSV log file."""
    os.makedirs(os.path.dirname(LOGS_PATH), exist_ok=True)

    file_exists = os.path.isfile(LOGS_PATH)
    with open(LOGS_PATH, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["timestamp", "prompt", "label",
                        "category", "confidence", "decision"]
        )
        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "prompt":     prompt[:120],   # truncate very long prompts
            "label":      label,
            "category":   category,
            "confidence": round(confidence, 4),
            "decision":   decision,
        })


def format_result(label: str, category: str,
                  confidence: float, decision: str) -> str:
    """Return a pretty-printed summary string for console output."""
    bar = "█" * int(confidence * 20) + "░" * (20 - int(confidence * 20))
    return (
        f"\n{'─'*45}\n"
        f"  Label      : {label}\n"
        f"  Category   : {category}\n"
        f"  Confidence : {confidence:.2%}  [{bar}]\n"
        f"  Decision   : {decision}\n"
        f"{'─'*45}\n"
    )
