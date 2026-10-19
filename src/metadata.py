"""
metadata.py — Extract and normalize ID3/Vorbis metadata from audio files.

Generates a stable track_id for each file using a BLAKE2b digest of the
first 1 MB of raw audio bytes. This ID is independent of mutable tags and
persists across metadata edits, making it safe to use as a join key across
the entire pipeline.

Output: metadata.csv
"""

import csv
import hashlib
import os
import re
from pathlib import Path

from mutagen._file import File as MutagenFile
from tqdm import tqdm

import config
from utils.logging_setup import get_logger
from utils.text import clean_string

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# CSV schema — column order is fixed for downstream consumers
# ---------------------------------------------------------------------------

FIELDS = [
    "track_id", "title", "track_number", "disc_number", "duration_seconds",
    "album", "album_artist", "contributing_artists", "genre_tagged", "year",
    "file_path",
]


# ---------------------------------------------------------------------------
# Tag helpers
# ---------------------------------------------------------------------------

def _safe_get(tags, keys: list[str]) -> str | None:
    """Try each key in order, returning the first non-empty value."""
    for key in keys:
        value = tags.get(key)
        if not value:
            continue
        raw = value[0] if isinstance(value, list) else value
        if hasattr(raw, "text"):
            raw = raw.text[0] if raw.text else ""
        return clean_string(raw)
    return None


def _parse_track_or_disc(value, default: int = 1) -> int:
    """Parse '3/12' → 3, handling missing or malformed values."""
    try:
        return int(str(value).split("/")[0]) if value else default
    except Exception:
        return default


def _parse_year(value) -> int | None:
    """Extract a 4-digit year from a date string (e.g. '2019-03-15' → 2019)."""
    try:
        return int(str(value)[:4]) if value else None
    except Exception:
        return None


def _parse_artists(raw: str | None) -> list[str] | None:
    """
    Split a combined artist string into a list.

    Handles separators such as:
    ; , & + × and featuring variants.
    """
    if not raw:
        return None

    raw = re.sub(r"\s*\b(ft\.?|feat\.?|featuring)\b\.?\s*", ";", raw, flags=re.IGNORECASE)

    raw = re.sub(r"\s+\band\b\s+", ";", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s+\bwith\b\s+", ";", raw, flags=re.IGNORECASE)

    artists = []

    for part in re.split(r"[;,&+×]", raw):
        artist = clean_string(part)

        if artist:
            artists.append(artist)

    return artists or None


def _parse_genres(raw: str | None) -> list[str] | None:
    """
    Parse comma-separated genres while preserving original formatting
    and duplicates.
    """
    if not raw:
        return None

    genres = [item.strip() for item in raw.split(",") if item.strip()]
    return genres or None


# ---------------------------------------------------------------------------
# Track ID generation
# ---------------------------------------------------------------------------

def generate_track_id(file_path: str) -> str:
    """
    Compute a BLAKE2b-12 hex digest of the first 1 MB of the audio file.

    Using raw bytes (not tags) ensures the ID is stable across tag edits.
    1 MB is enough to make collisions negligibly unlikely for a 25k catalog.
    """
    hasher = hashlib.blake2b(digest_size=6)
    try:
        with open(file_path, "rb") as f:
            hasher.update(f.read(1_048_576))
    except Exception as e:
        log.warning("Could not read %s for ID generation, falling back to path hash: %s", file_path, e)
        hasher.update(file_path.encode("utf-8"))
    return hasher.hexdigest()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def extract_metadata() -> list[dict]:
    """
    Walk the raw audio directory, parse tags, and build one row per track.

    Duplicate track IDs (hash collisions) are resolved by appending a short
    path-based suffix so every row has a unique ID.
    """
    audio_files = sorted(
        str(p)
        for p in Path(config.RAW_AUDIO_DIR).rglob("*")
        if p.suffix.lower() in config.AUDIO_EXTENSIONS
    )
    log.info("Found %d audio files in %s", len(audio_files), config.RAW_AUDIO_DIR)

    rows: list[dict] = []
    seen_ids: dict[str, str] = {}   # id → first path (collision detection)
    ok = missed = 0

    for path in tqdm(audio_files, desc="Metadata scan", dynamic_ncols=True):
        try:
            audio = MutagenFile(path, easy=False)
            if not audio or not getattr(audio, "info", None):
                tqdm.write(f"[WARN] Skipping malformed audio header: {path}")
                missed += 1
                continue

            tags    = audio.tags or {}
            artists = _parse_artists(_safe_get(tags, ["TPE1", "artist"]))
            genres  = _parse_genres(_safe_get(tags, ["TCON", "genre"]))
            track_id = generate_track_id(path)

            if track_id in seen_ids:
                tqdm.write(f"[WARN] Hash collision {track_id}: {seen_ids[track_id]} vs {path}")
                suffix   = hashlib.blake2b(path.encode("utf-8"), digest_size=2).hexdigest()
                track_id = f"{track_id}_{suffix}"

            seen_ids[track_id] = path
            rows.append({
                "track_id":             track_id,
                "title":                clean_string(_safe_get(tags, ["TIT2", "title"])),
                "track_number":         _parse_track_or_disc(_safe_get(tags, ["TRCK", "tracknumber"])),
                "disc_number":          _parse_track_or_disc(_safe_get(tags, ["TPOS", "discnumber"]), 1),
                "duration_seconds":     int(round(audio.info.length)),
                "album":                clean_string(_safe_get(tags, ["TALB", "album"])),
                "album_artist":         clean_string(_safe_get(tags, ["TPE2", "albumartist"])),
                "contributing_artists": ";".join(artists) if artists else None,
                "genre_tagged":         ";".join(genres)  if genres  else None,
                "year":                 _parse_year(_safe_get(tags, ["TDRC", "date", "YEAR"])),
                "file_path":            path,
            })
            ok += 1

        except Exception as e:
            tqdm.write(f"[ERROR] {path}: {e}")
            missed += 1

    log.info("Extraction complete — valid: %d, skipped: %d", ok, missed)
    return rows


def write_csv(rows: list[dict]) -> None:
    """Write extracted rows to metadata.csv, creating directories as needed."""
    config.EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.METADATA_CSV, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (row.get(k) if row.get(k) is not None else "") for k in FIELDS})
    log.info("Wrote %d rows → %s", len(rows), config.METADATA_CSV)


if __name__ == "__main__":
    write_csv(extract_metadata())