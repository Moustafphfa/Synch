"""
lyrics.py — Fetch lyrics from the Genius API and scrape the Genius HTML page.

Run with:
    python src/lyrics.py --token YOUR_GENIUS_TOKEN --workers 3

The script is resumable: already-fetched tracks (those with lyrics or a
definitive failure reason) are skipped on re-runs. Retryable failures
(network errors, transient issues) are re-attempted automatically.

Output: lyrics.jsonl, missed_lyrics.csv
"""

import argparse
import json
import os
import re
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from tqdm import tqdm

import config
from utils.logging_setup import get_logger
from utils.text import normalize_for_matching, token_overlap

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Title / artist normalization
# ---------------------------------------------------------------------------

def _primary_artist(artist: str) -> str:
    """Extract the first-billed artist from a collaborative credit string."""
    return re.split(
        r"\s*[&+]\s*|\s*(feat\.?|ft\.?|featuring)\s*",
        artist, flags=re.IGNORECASE,
    )[0].strip()


def _clean_title_for_search(title: str) -> str:
    """Strip parentheticals and release variants that confuse the search API."""
    title = re.sub(r"\s*[\(\[].*?[\)\]]", "", title)
    title = re.sub(
        r"\s*-\s*(single|album|remaster.*|radio edit|instrumental|live.*|demo)$",
        "", title, flags=re.IGNORECASE,
    )
    return re.sub(r"[^\w\s\-']", " ", title).strip()


def _split_medley(title: str) -> list[str]:
    """Split 'Song A / Song B' medley titles into individual search strings."""
    return [p.strip() for p in re.split(r"\s*/\s*", title) if p.strip()] if "/" in title else []


def _is_emoji_only(title: str) -> bool:
    """Return True if the title contains no alphanumeric characters."""
    return not re.sub(r"[^\w\s]", "", title).strip()


def _is_interlude(title: str) -> bool:
    """Return True for transition tracks that warrant a lower match threshold."""
    return any(k in title.lower() for k in
               ["interlude", "intro", "outro", "skit", "reprise", "transition"])


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------

def _build_queries(title: str, artist: str, album: str | None) -> list[str]:
    """
    Build a prioritised list of search query strings.

    Tries album-qualified, artist-qualified, and plain title variants, plus
    medley splits for tracks with slashes in their names.
    """
    clean_title = _clean_title_for_search(title)
    primary     = _primary_artist(artist)
    queries: list[str] = []

    for part in _split_medley(clean_title):
        queries += [f"{part} {primary}", f"{part} {artist}"]

    if album:
        queries.append(f"{clean_title} {primary} {album}")

    queries += [f"{clean_title} {primary}", f"{clean_title} {artist}", clean_title]

    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for q in queries:
        q = re.sub(r"\s+", " ", q).strip()
        if q and q not in seen:
            seen.add(q)
            out.append(q)
    return out


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------

def _score_candidate(
    result: dict,
    title: str,
    artist: str,
    album: str | None,
    year: int | None,
    track_number: int | None,
) -> int:
    """
    Score a Genius search result against the local track metadata.

    Higher score = better match. Weights:
      title match  50 (exact) / up to 30 (partial)
      artist match 25 (strong) / 12 (weak)
      album match  15 / 7
      year match    5 / 3 (off-by-one)
      track number  5
    """
    score = 0

    h_title = normalize_for_matching(result.get("title", ""))
    o_title = normalize_for_matching(title)
    score += 50 if h_title == o_title else int(token_overlap(h_title, o_title) * 30)

    h_artist = normalize_for_matching(result.get("primary_artist", {}).get("name", ""))
    overlap  = token_overlap(h_artist, normalize_for_matching(_primary_artist(artist)))
    score += 25 if overlap >= 0.5 else (12 if overlap >= 0.2 else 0)

    if album:
        h_album = normalize_for_matching((result.get("album") or {}).get("name", ""))
        if h_album:
            ov = token_overlap(normalize_for_matching(album), h_album)
            score += 15 if ov >= 0.5 else (7 if ov >= 0.25 else 0)

    if year:
        h_year = (result.get("release_date_components") or {}).get("year")
        if h_year:
            diff = abs(int(h_year) - int(year))
            score += 5 if diff == 0 else (3 if diff == 1 else 0)

    if track_number:
        h_track = (result.get("album") or {}).get("track_number")
        if h_track and int(h_track) == int(track_number):
            score += 5

    return score


# ---------------------------------------------------------------------------
# Genius API client
# ---------------------------------------------------------------------------

class GeniusClient:
    """Thread-safe HTTP client for the Genius API and lyric scraping."""

    BASE_URL = "https://api.genius.com"

    def __init__(self, token: str, workers: int = config.LYRICS_DEFAULT_WORKERS):
        # Rate-limit delay is distributed across workers
        self._delay        = workers / config.LYRICS_TARGET_RPS
        self._empty_streak = 0
        self._lock         = Lock()

        self.session = requests.Session()
        # Match HTTP connection pool size to thread count to avoid exhaustion
        adapter = HTTPAdapter(pool_connections=workers, pool_maxsize=workers)
        self.session.mount("https://", adapter)
        self.session.mount("http://",  adapter)
        self.session.headers.update({
            "Authorization": f"Bearer {token}",
            "User-Agent":    "Synch/1.0",
        })

    def verify_token(self) -> None:
        """Make a test request to fail fast on bad credentials."""
        r = self.session.get(f"{self.BASE_URL}/search", params={"q": "Kanye West"}, timeout=10)
        if r.status_code == 403:
            raise SystemExit("[FATAL] Genius token rejected (403). Pass a valid token via --token.")
        r.raise_for_status()
        if not r.json()["response"].get("hits"):
            raise SystemExit("[FATAL] Token accepted but search returned no results. Check app permissions.")
        log.info("Genius token verified. Per-request delay: %.2fs", self._delay)

    def search(self, query: str) -> list:
        """Execute a search query with retry and shadow-ban protection."""
        for attempt in range(4):
            try:
                r = self.session.get(
                    f"{self.BASE_URL}/search", params={"q": query}, timeout=10
                )

                if r.status_code == 403:
                    raise SystemExit("[FATAL] Genius token revoked mid-run.")

                if r.status_code == 429:
                    wait = 30 * (attempt + 1)
                    tqdm.write(f"[WARN] 429 rate limit — waiting {wait}s")
                    time.sleep(wait)
                    continue

                r.raise_for_status()
                hits = r.json()["response"].get("hits", [])

                with self._lock:
                    self._empty_streak = 0 if hits else self._empty_streak + 1
                    streak = self._empty_streak

                if streak >= 5:
                    tqdm.write("[WARN] 5 consecutive empty results — possible shadow ban. Pausing 60s.")
                    time.sleep(60)
                    with self._lock:
                        self._empty_streak = 0

                time.sleep(self._delay)
                return hits

            except SystemExit:
                raise
            except Exception as e:
                tqdm.write(f"[ERR] Search failed for {query!r}: {e}")
                time.sleep(5 * (attempt + 1))

        return []

    def scrape_lyrics(self, url: str) -> tuple[str | None, str]:
        """Scrape the lyrics text from a Genius song page."""
        try:
            r = self.session.get(url, timeout=15)
            r.raise_for_status()
        except Exception:
            return None, "request_failed"

        soup = BeautifulSoup(r.text, "html.parser")
        containers = (
            soup.find_all("div", {"data-lyrics-container": "true"})
            or soup.find_all("div", class_=re.compile("Lyrics__Container"))
        )

        if not containers:
            return None, "no_lyrics_on_genius"

        lines = []
        for c in containers:
            for br in c.find_all("br"):
                br.replace_with("\n")
            lines.append(c.get_text("\n"))

        text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
        return (text, "ok") if text else (None, "no_lyrics_on_genius")


# ---------------------------------------------------------------------------
# Per-track processing
# ---------------------------------------------------------------------------

def _process_track(client: GeniusClient, row: dict) -> dict:
    """Fetch lyrics for one track. Returns a result dict with lyrics and metadata."""
    track_id = str(row["track_id"])
    title    = str(row.get("title") or "").strip()
    artist   = str(row.get("album_artist") or row.get("contributing_artists") or "").split(";")[0].strip()
    album    = str(row.get("album") or "").strip() or None
    year     = int(row["year"])         if pd.notna(row.get("year"))         else None
    tn       = int(row["track_number"]) if pd.notna(row.get("track_number")) else None
    duration = int(row.get("duration_seconds") or 0)

    def _result(lyrics, reason, score=0):
        return {"track_id": track_id, "lyrics": lyrics, "score": score, "reason": reason}

    if not title or not artist:
        return _result(None, "missing_metadata")
    if duration < config.LYRICS_SHORT_TRACK_SEC:
        return _result(None, "short_track")
    if _is_emoji_only(title):
        return _result(None, "emoji_title")

    threshold    = config.LYRICS_LOW_THRESHOLD if _is_interlude(title) else config.LYRICS_MATCH_THRESHOLD
    best_hit     = None
    best_score   = 0
    got_any_hits = False

    for query in _build_queries(title, artist, album):
        hits = client.search(query)
        if hits:
            got_any_hits = True
        for hit in hits:
            candidate = hit.get("result", {})
            s = _score_candidate(candidate, _clean_title_for_search(title), artist, album, year, tn)
            if s > best_score:
                best_hit, best_score = candidate, s
        if best_score >= threshold:
            break

    if not best_hit or best_score < threshold:
        return _result(None, "no_match" if got_any_hits else "no_search_results", best_score)

    lyrics, reason = client.scrape_lyrics(best_hit["url"])
    return _result(lyrics, reason, best_score)


# ---------------------------------------------------------------------------
# JSONL state management
# ---------------------------------------------------------------------------

def _load_completed_ids() -> set[str]:
    """
    Return track IDs that already have a definitive outcome.

    Definitive = has lyrics, or a known-terminal failure (e.g. no_lyrics_on_genius).
    Non-definitive outcomes (network errors, etc.) are re-attempted.
    """
    done: set[str] = set()
    retryable = 0

    if not config.LYRICS_JSONL.exists():
        return done

    with open(config.LYRICS_JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                if rec.get("lyrics") or rec.get("match_reason") in config.LYRICS_DEFINITIVE_OUTCOMES:
                    done.add(rec["track_id"])
                else:
                    retryable += 1
            except Exception:
                pass

    if retryable:
        log.info("%d previously failed tracks will be retried", retryable)

    return done


def _append_result(rec: dict, lock: Lock) -> None:
    """Append one result to the JSONL file. Thread-safe."""
    with lock:
        with open(config.LYRICS_JSONL, "a", encoding="utf-8") as f:
            json.dump({
                "track_id":     rec["track_id"],
                "lyrics":       rec["lyrics"],
                "match_score":  rec["score"],
                "match_reason": rec["reason"],
            }, f, ensure_ascii=False)
            f.write("\n")


def _compact_jsonl() -> int:
    """Deduplicate the JSONL file, keeping the last entry per track_id."""
    records: dict[str, str] = {}
    with open(config.LYRICS_JSONL, encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                records[rec["track_id"]] = line
            except Exception:
                pass
    with open(config.LYRICS_JSONL, "w", encoding="utf-8") as f:
        f.writelines(records.values())
    return len(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Genius lyrics for all tracks in metadata.csv.")
    parser.add_argument("--token",   required=True, help="Genius API OAuth token")
    parser.add_argument("--workers", type=int, default=config.LYRICS_DEFAULT_WORKERS,
                        help=f"Thread count (default: {config.LYRICS_DEFAULT_WORKERS}; keep ≤ 5)")
    args = parser.parse_args()

    config.EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    client = GeniusClient(args.token, workers=args.workers)
    client.verify_token()

    metadata    = pd.read_csv(config.METADATA_CSV)
    done        = _load_completed_ids()
    pending     = [r for r in metadata.to_dict("records") if str(r["track_id"]) not in done]
    write_lock  = Lock()

    log.info("Tracks total: %d | already done: %d | pending: %d",
             len(metadata), len(done), len(pending))

    fetched     = 0
    missed_rows: list[dict] = []

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_track, client, row): row for row in pending}

        with tqdm(total=len(pending), desc="Lyrics fetch", dynamic_ncols=True) as bar:
            for n, future in enumerate(as_completed(futures), 1):
                row = futures[future]
                try:
                    rec = future.result()
                except Exception as e:
                    rec = {"track_id": str(row["track_id"]), "lyrics": None,
                           "score": 0, "reason": f"error:{e}"}

                _append_result(rec, write_lock)

                if rec["lyrics"]:
                    fetched += 1
                else:
                    missed_rows.append({
                        "track_id":     rec["track_id"],
                        "title":        row.get("title"),
                        "artist":       row.get("album_artist") or row.get("contributing_artists"),
                        "album":        row.get("album"),
                        "year":         row.get("year"),
                        "track_number": row.get("track_number"),
                        "duration_sec": row.get("duration_seconds"),
                        "reason":       rec["reason"],
                        "best_score":   rec["score"],
                    })
                    tqdm.write(
                        f"[{rec['reason']:22s}] score={rec['score']:2d}  "
                        f"{str(row.get('title'))!r} → "
                        f"{row.get('album_artist') or row.get('contributing_artists')}"
                    )

                bar.set_postfix(fetched=fetched, missed=len(missed_rows))
                bar.update(1)

                if n % config.LYRICS_BATCH_SIZE == 0:
                    tqdm.write(f"[CHECKPOINT] {n} tracks processed: fetched={fetched}, missed={len(missed_rows)}")

    unique = _compact_jsonl()
    log.info("Done — fetched: %d, missed: %d, unique JSONL entries: %d",
             fetched, len(missed_rows), unique)

    if missed_rows:
        pd.DataFrame(missed_rows).to_csv(config.MISSED_LYRICS_CSV, index=False)
        log.info("Missed tracks written to %s", config.MISSED_LYRICS_CSV)


if __name__ == "__main__":
    main()