# threshold.py — Applies a configurable confidence threshold to decide
#               whether a prompt should be allowed through or blocked.

from app.config import THRESHOLD, LABEL_UNSAFE


def apply_threshold(label: str, confidence: float,
                    threshold: float = THRESHOLD) -> str:
    """
    Return 'BLOCK' or 'ALLOW' based on label and confidence.

    Rules
    -----
    - Unsafe AND confidence > threshold  → BLOCK
    - Unsafe AND confidence <= threshold → ALLOW  (low-confidence unsafe)
    - Safe  (any confidence)             → ALLOW
    """
    if label == LABEL_UNSAFE and confidence > threshold:
        return "BLOCK"
    return "ALLOW"


def adjust_threshold(new_value: float) -> float:
    """Validate and return a new threshold value (must be 0–1)."""
    if not 0.0 <= new_value <= 1.0:
        raise ValueError(f"Threshold must be between 0 and 1, got {new_value}")
    return new_value
