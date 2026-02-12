import os
import csv
import re
import unicodedata
import logging
from pathlib import Path

import ftfy
from mutagen import File as MutagenFile


INPUT_DIR = "Path"
OUTPUT_DIR = "Path"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")

AUDIO_EXTS = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".aac"}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def clean_string(s: str | None) -> str | None: 
    # Normalizes tag text across encodings and Unicode variants.
    if not s:
        return None

    s = ftfy.fix_text(str(s))
    s = unicodedata.normalize("NFC", s)

    s = re.sub(r"[‐-–—−]", "-", s)
    s = s.replace("'", "'").replace("'", "'").replace(""", '"').replace(""", '"')
    s = re.sub(r"[\u200B-\u200D\uFEFF]", "", s)
    s = s.replace("\r", "")

    return s.strip()


def safe_get(tags: dict, keys: list[str]) -> str | None:
    for k in keys:
        if k in tags:
            val = tags[k]
            if isinstance(val, list):
                return clean_string(val[0])
            return clean_string(val)
    return None


def parse_track_or_disc(value, default=1) -> int:
    if not value:
        return default
    try:
        return int(str(value).split("/")[0])
    except Exception:
        return default


def parse_year(value) -> int | None:
    if not value:
        return None
    try:
        return int(str(value)[:4])
    except Exception:
        return None


def parse_artists(raw: str | None) -> list[str] | None:
    # Normalizes artist strings and splits on common separators.
    if not raw:
        return None

    raw = clean_string(raw)

    raw = re.sub(
        r"\b(ft\.?|feat\.?|featuring)\b\.?",
        ";",
        raw,
        flags=re.IGNORECASE,
    )

    artists = re.split(r"[;,/&]", raw)
    artists = [clean_string(a) for a in artists if a.strip()]

    return artists or None


def parse_genres(raw: str | None) -> list[str] | None:
    # Splits genre tags while preserving semantic slashes.
    if not raw:
        return None

    raw = clean_string(raw)

    split_genres = []
    for g in re.split(r"[;,]", raw):
        split_genres.extend([s.strip() for s in g.split(" / ")])

    seen = set()
    deduped = []
    for g in split_genres:
        gl = g.lower()
        if gl not in seen:
            deduped.append(g)
            seen.add(gl)

    return deduped or None


def sanitize_id(s: str) -> str:
    return re.sub(r"[^a-z0-9_-]", "", s.lower())


def generate_track_id(row: dict) -> str:
    # Deterministic ID based on legacy tagging logic.
    def first_char(v): return v[0] if v else "X"
    def last_char(v): return v[-1] if v else "X"

    raw_id = (
        f"{first_char(row.get('title'))}"
        f"{row.get('track_number', 0)}"
        f"{row.get('disc_number', 1)}"
        f"{str(row.get('duration_seconds', 0))[:2]}"
        f"{first_char(row.get('album'))}"
        f"{first_char(row.get('album_artist'))}"
        f"{last_char(row.get('contributing_artists'))}"
        f"{first_char(row.get('genre_tagged'))}"
        f"{str(row.get('year', 'XX'))[-2:]}"
    )

    return sanitize_id(raw_id)


def extract_metadata() -> list[dict]:
    rows = []
    processed = 0

    for root, _, files in os.walk(INPUT_DIR):
        for fname in files:
            if Path(fname).suffix.lower() not in AUDIO_EXTS:
                continue

            path = os.path.join(root, fname)

            try:
                audio = MutagenFile(path, easy=False)
                if not audio or not hasattr(audio, "info"):
                    continue

                tags = audio.tags or {}

                artists_list = parse_artists(safe_get(tags, ["TPE1", "artist"]))
                genres_list = parse_genres(safe_get(tags, ["TCON", "genre"]))

                row = {
                    "title": safe_get(tags, ["TIT2", "title"]),
                    "track_number": parse_track_or_disc(safe_get(tags, ["TRCK", "tracknumber"])),
                    "disc_number": parse_track_or_disc(safe_get(tags, ["TPOS", "discnumber"]), 1),
                    "duration_seconds": int(round(audio.info.length)),
                    "album": safe_get(tags, ["TALB", "album"]),
                    "album_artist": safe_get(tags, ["TPE2", "albumartist"]),
                    "contributing_artists": ";".join(artists_list) if artists_list else None,
                    "genre_tagged": ";".join(genres_list) if genres_list else None,
                    "year": parse_year(safe_get(tags, ["TDRC", "date", "YEAR"])),
                    "file_path": path,
                }

                row["track_id"] = generate_track_id(row)
                rows.append(row)
                processed += 1

                if processed % 500 == 0:
                    logging.info("Processed %d tracks", processed)

            except Exception:
                logging.exception("Failed processing file: %s", path)

    logging.info("Finished extraction: %d tracks", processed)
    return rows


def write_csv(rows: list[dict]):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fieldnames = [
        "track_id",
        "title",
        "track_number",
        "disc_number",
        "duration_seconds",
        "album",
        "album_artist",
        "contributing_artists",
        "genre_tagged",
        "year",
        "file_path"
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            # CSV requires empty fields rather than None.
            safe_row = {k: (r.get(k) if r.get(k) is not None else "") for k in fieldnames}
            writer.writerow(safe_row)


if __name__ == "__main__":
    data = extract_metadata()
    write_csv(data)
    logging.info("Extracted metadata for %d tracks → %s", len(data), OUTPUT_CSV)