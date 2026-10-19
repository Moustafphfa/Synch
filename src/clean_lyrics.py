"""
cleanlyrics.py — Parse and clean raw scraped lyrics.

Strips non-lyrical noise (contributor counts, "Read More" banners, title
headers, inline annotations) and normalises structural tags like
[Verse 1: Drake] → <VERSE> for consistent downstream NLP tokenisation.

Tracks without usable lyrics (null payloads, instrumentals, or content that
reduces to nothing after cleaning) are written to missed_cleanlyrics.jsonl
with a reason code. Their entry in cleanlyrics.jsonl has lyrics: null.

Input:  lyrics.jsonl
Output: cleanlyrics.jsonl, missed_cleanlyrics.jsonl, dead_letter.jsonl
"""

import json
import os
import re
import unicodedata
from collections import Counter
from pathlib import Path

from tqdm import tqdm

import config
from utils.logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# JSONL loading with repair
# ---------------------------------------------------------------------------

def _repair_json(raw: str) -> dict | None:
    """
    Attempt to parse a potentially malformed JSON string.

    Tier 1: standard parse.
    Tier 2: escaped-quote fix.
    Tier 3: regex extraction of known field layout.
    """
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    try:
        return json.loads(raw.replace('""', '\\"'))
    except json.JSONDecodeError:
        pass

    m = re.match(
        r'\{"track_id":\s*"([^"]+)",\s*"lyrics":\s*"(.*)",\s*"match_score":\s*(\d+),\s*"match_reason":\s*"([^"]+)"\s*\}',
        raw, re.DOTALL,
    )
    if m:
        return {
            "track_id":    m.group(1),
            "lyrics":      m.group(2),
            "match_score": int(m.group(3)),
            "match_reason": m.group(4),
        }
    return None


def _load_jsonl(path: str) -> tuple[list[dict], list[dict]]:
    """
    Load the lyrics JSONL file, handling multi-line records.

    Returns (records, dead_letters). Dead letters are unfixable chunks that
    are written to dead_letter.jsonl for manual inspection.
    """
    content = Path(path).read_text(encoding="utf-8")
    # Each record starts with {"track_id" — split on that boundary
    chunks = re.split(r'(?=\{"track_id")', content)

    records:      list[dict] = []
    dead_letters: list[dict] = []

    for chunk in chunks:
        raw = chunk.strip()
        if not raw:
            continue
        obj = _repair_json(raw.replace("\n", " ").replace("\r", " "))
        if obj is not None:
            records.append(obj)
        else:
            dead_letters.append({"type": "repair_failure", "buffer": raw[:500]})

    log.info("Loaded %d records; %d unrecoverable", len(records), len(dead_letters))
    return records, dead_letters


# ---------------------------------------------------------------------------
# Noise removal regexes
# ---------------------------------------------------------------------------

_RE_CONTRIBUTORS = re.compile(r"^\d+\s+Contributors?\s*\n",           re.IGNORECASE)
_RE_TRANSLATIONS = re.compile(r"^Translations?\s*\n(?:[^\[{}\n]+\n)*", re.IGNORECASE)
_RE_TITLE_HEADER = re.compile(r"^.{1,120} Lyrics\s*\n")
_RE_SONG_DESC    = re.compile(r"^.{10,500}?\u2026?\s*\nRead More\s*\n", re.DOTALL | re.IGNORECASE)
_RE_READ_MORE    = re.compile(r"\nRead More\s*\n",                       re.IGNORECASE)
_RE_PRODUCED_BY  = re.compile(r"\[Produced by[^\]]*\]\s*",             re.IGNORECASE)
_RE_INLINE_ANN   = re.compile(r"\n[ \t]+([^\n]+)\n[ \t]+")
_RE_INSTRUMENTAL = re.compile(r"^\s*this\s+song\s+is\s+an\s+instrumental\s*$",
                               re.IGNORECASE | re.MULTILINE)
_RE_NBSP         = re.compile(r"\xa0")
_RE_INVISIBLE    = re.compile(r"[\u200b\u200c\u200d\u200e\u200f\u2060\ufeff]")

_QUOTE_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"', "\u201f": '"',
    "\u2039": "'", "\u203a": "'", "\u00ab": '"', "\u00bb": '"',
}
_RE_QUOTES = re.compile("[" + "".join(_QUOTE_MAP.keys()) + "]")


# ---------------------------------------------------------------------------
# Cleaning pipeline
# ---------------------------------------------------------------------------

def _normalise_structural_tags(text: str) -> str:
    """
    Replace Genius section labels with canonical XML-style tags.

    Examples:
        [Chorus]              → <CHORUS>
        [Verse 1: Drake]      → <VERSE>
        [Bridge]              → <BRIDGE>
        [Hook: Lil Wayne]     → <HOOK>   (unknown tags preserve their name)
    """
    def _replace(match):
        tag = match.group(1).upper()
        for keyword, replacement in [
            ("CHORUS", "<CHORUS>"),
            ("VERSE",  "<VERSE>"),
            ("BRIDGE", "<BRIDGE>"),
            ("INTRO",  "<INTRO>"),
            ("OUTRO",  "<OUTRO>"),
        ]:
            if keyword in tag:
                return f"\n{replacement}\n"
        # Unknown tag: wrap the normalised content
        label = re.sub(r":.*", "", tag, flags=re.DOTALL).strip()  # drop artist attribution (incl. multi-line)
        return f"\n<{label}>\n"

    return re.sub(r"\[(.*?)\]", _replace, text, flags=re.DOTALL)


def _clean_lyrics(text: str) -> str:
    """Apply the full sequential cleaning pipeline to a raw lyrics string."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _RE_NBSP.sub(" ", text)

    text = _RE_CONTRIBUTORS.sub("", text)
    text = _RE_TRANSLATIONS.sub("", text)
    text = _RE_SONG_DESC.sub("", text)
    text = _RE_READ_MORE.sub("\n", text)
    text = _RE_TITLE_HEADER.sub("", text)
    text = _RE_PRODUCED_BY.sub("", text)

    # Normalise structure for NLP
    text = _normalise_structural_tags(text)

    # Fix inline annotation indentation artefacts
    text = _RE_INLINE_ANN.sub(lambda m: f" {m.group(1)} ", text)

    text = _RE_QUOTES.sub(lambda m: _QUOTE_MAP[m.group()], text)
    text = unicodedata.normalize("NFC", text)

    text = _RE_INVISIBLE.sub("", text)
    text = re.sub(r'\(\s*\n\s*([^)]*?)\s*\)', r'(\1)', text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    
    # -------------------------------------------------------------
    # NEW MODIFICATIONS APPLIED HERE
    # -------------------------------------------------------------
    
    # 1. Remove double \n\n (collapse all multiple newlines to a single \n)
    text = re.sub(r"\n{2,}", "\n", text)
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Keep tags capitalized (find <tag> and convert back to <TAG>)
    text = re.sub(r"<([a-z0-9_-]+)>", lambda m: f"<{m.group(1).upper()}>", text)

    return text.strip()


def _is_instrumental(raw: str) -> bool:
    return bool(_RE_INSTRUMENTAL.search(raw))


def clean_lyrics(raw: str) -> str | None:
    """
    Gate function: returns cleaned lyrics or None if there's nothing usable.

    None means the track should be recorded with a skip reason, not silently
    dropped.
    """
    if not raw or not raw.strip():
        return None
    text = _clean_lyrics(raw)
    text = _RE_INSTRUMENTAL.sub("", text).strip()
    return text or None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config.EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    log.info("Loading lyrics from %s", config.LYRICS_JSONL)
    records, dead_letters = _load_jsonl(str(config.LYRICS_JSONL))

    if dead_letters:
        with open(config.DEAD_LETTER_JSONL, "w", encoding="utf-8") as f:
            for r in dead_letters:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.warning("%d unrecoverable records written to %s", len(dead_letters), config.DEAD_LETTER_JSONL)

    ok = skipped = 0
    skip_reasons: list[str] = []

    with (
        open(config.CLEAN_LYRICS_JSONL, "w", encoding="utf-8") as out_f,
        open(config.MISSED_CLEAN_JSONL, "w", encoding="utf-8") as miss_f,
    ):
        for obj in tqdm(records, desc="Cleaning lyrics", dynamic_ncols=True):
            track_id = obj.get("track_id", "")
            raw_lyr  = obj.get("lyrics")

            skip_reason = None
            cleaned     = None

            if raw_lyr is None:
                skip_reason = "null_payload"
            elif _is_instrumental(raw_lyr):
                skip_reason = "is_instrumental"
            else:
                cleaned = clean_lyrics(raw_lyr)
                if not cleaned:
                    skip_reason = "reduced_to_empty"

            out_f.write(json.dumps(
                {"track_id": track_id, "lyrics": cleaned},
                ensure_ascii=False,
            ) + "\n")

            if skip_reason:
                miss_f.write(json.dumps(
                    {"track_id": track_id, "skip_reason": skip_reason},
                    ensure_ascii=False,
                ) + "\n")
                skip_reasons.append(skip_reason)
                skipped += 1
            else:
                ok += 1

    log.info("Done — with lyrics: %d, skipped: %d", ok, skipped)

    if skip_reasons:
        for reason, count in sorted(Counter(skip_reasons).items()):
            log.info("  skip reason %-25s %d", reason, count)


if __name__ == "__main__":
    main()