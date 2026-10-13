"""Text normalization for spoken command processing."""

import re

# Comprehensive spoken English contractions
CONTRACTIONS = {
    r"\bwhat's\b": "what is",
    r"\bhow's\b": "how is",
    r"\bit's\b": "it is",
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bdon't\b": "do not",
    r"\bi'm\b": "i am",
    r"\bi'd\b": "i would",
    r"\bi'll\b": "i will",
    r"\byou'll\b": "you will",
    r"\blet's\b": "let us",
    r"\bthere's\b": "there is",
    r"\bcouldn't\b": "could not",
    r"\bshouldn't\b": "should not",
    r"\bwouldn't\b": "would not",
    r"\bdidn't\b": "did not",
    r"\bdoesn't\b": "does not",
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
}


def normalize_text(text: str) -> str:
    """Normalizes raw transcribed speech into a clean, canonical string.

    Steps:
    1. Lowercase string
    2. Expand spoken contractions
    3. Remove punctuation (retaining alphanumeric, spaces, and commas for segmentation if requested)
    4. Collapse consecutive whitespaces and trim edges
    """
    if not text:
        return ""

    lowered = text.strip().lower()

    for pattern, replacement in CONTRACTIONS.items():
        lowered = re.sub(pattern, replacement, lowered)

    # Remove all punctuation and symbols except letters, numbers, and spaces
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)

    # Collapse repeated whitespace
    collapsed = re.sub(r"\s+", " ", cleaned).strip()
    return collapsed


def strip_polite_fillers(text: str) -> str:
    """Strips common polite prefixes, suffixes, and wake/filler words.

    Examples:
    "please turn on the lights" -> "turn on the lights"
    "could you turn on the AC please" -> "turn on the ac"
    "hey assistant turn off the lights" -> "turn off the lights"
    """
    cleaned = text
    # Prefixes
    cleaned = re.sub(
        r"^(hey |hi |hello )?(assistant|car|vehicle|drivesense|computer)[,\s]+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"^(please |could you please |can you please |would you please |could you |can you |would you |i want |i would like to |kindly )",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    # Suffixes
    cleaned = re.sub(
        r"[\s,]+(please|for me|right now|thank you|thanks)$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()
