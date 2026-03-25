# baseline.py — keyword-only baseline classifier


import re

# Flat keyword list — no category granularity, just safe/unsafe
_UNSAFE_PATTERNS = [
    # jailbreak
    r"\bdan\b", r"do anything now", r"jailbreak", r"ignore.*instructions?",
    r"pretend.*no restrictions?", r"act as.*evil", r"bypass.*safety",
    r"no restrictions?", r"forget.*training",
    # injection
    r"system prompt", r"override.*system", r"new instruction",
    r"disregard.*previous", r"###\s*instruction",
    # toxic
    r"hate.*\b(race|religion|ethnic|gender)\b",
    r"(slur|dehumanise|harass|bully)",
    r"incite.*violence",
    # harmful
    r"how.*(make|build|create).*(bomb|explosive|weapon|poison)",
    r"(synthesise|synthesize|manufacture).*(drug|meth|fentanyl)",
    r"hack.*database", r"ransomware", r"suicide.*method",
    r"(buy|obtain).*illegal.*firearm",
    r"groom.*minor", r"ddos.*attack",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _UNSAFE_PATTERNS]


def baseline_classify(prompt: str) -> dict:
    """
    Returns a simple dict with label, category='unknown', and confidence.
    Category is always 'unknown' — the baseline does not distinguish types.
    """
    prompt_lower = prompt.lower()
    for pattern in _COMPILED:
        if pattern.search(prompt_lower):
            return {"label": "Unsafe", "category": "unknown", "confidence": 0.85}

    return {"label": "Safe", "category": "safe", "confidence": 0.80}
