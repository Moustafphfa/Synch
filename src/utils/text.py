"""
utils/text.py — Shared text normalization utilities.

Previously duplicated across metadata.py and lyrics.py.
Centralised here to keep a single source of truth.
"""

import re
import unicodedata

# Module-level import to avoid overhead in hot loops
try:
    import ftfy
    HAS_FTFY = True
except ImportError:
    HAS_FTFY = False

# Matches invisible / zero-width / BOM characters that corrupt downstream NLP
_INVISIBLE = re.compile(
    r"[\u00AD"         # soft hyphen
    r"\u200B-\u200D"   # zero-width space / non-joiner / joiner
    r"\u2060"          # word joiner
    r"\uFEFF"          # BOM / zero-width no-break space
    r"\uFFFD]"         # replacement character (encoding error)
)


def coerce_str(value) -> str:
    """Decode bytes to str, with a latin-1 fallback for broken encodings."""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("latin-1")
    return str(value)


def clean_string(value) -> str | None:
    """
    Multi-stage Unicode normalization.

    Pipeline: coerce → mojibake repair (ftfy) → NFC → strip invisibles → trim.
    Returns None for empty or whitespace-only inputs.
    """
    if not value and value != 0:
        return None
    s = coerce_str(value)
    if not s.strip():
        return None

    if HAS_FTFY:
        s = ftfy.fix_text(s)

    s = unicodedata.normalize("NFC", s)
    s = _INVISIBLE.sub("", s)
    s = s.replace("\r", "").strip()
    return s or None


def normalize_for_matching(s: str) -> str:
    """
    Lowercase, strip diacritics, and remove non-alphanumeric characters.

    Used for fuzzy title/artist matching in the lyrics scoring step.
    """
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower())).strip()


def token_overlap(a: str, b: str) -> float:
    """Jaccard similarity over whitespace-tokenized sets."""
    ta, tb = set(a.split()), set(b.split())
    return len(ta & tb) / len(ta | tb) if (ta and tb) else 0.0