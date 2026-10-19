"""
config.py — Central configuration for the Synch pipeline.

All paths, constants, and tuneable parameters live here.
Override the base directory via the SYNCH_BASE environment variable.

    SYNCH_BASE=/mnt/external/synch python metadata.py
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------

BASE_DIR      = Path(os.getenv("SYNCH_BASE", "somewhere"))
DATA_DIR      = BASE_DIR / "data"
RAW_AUDIO_DIR = DATA_DIR / "raw audio"
EXTRACT_DIR   = DATA_DIR / "extractions"
MODEL_DIR     = BASE_DIR / "models"
EMBED_CACHE_DIR = EXTRACT_DIR / "embed_cache"

# ---------------------------------------------------------------------------
# Pipeline output files
# ---------------------------------------------------------------------------

METADATA_CSV       = EXTRACT_DIR / "metadata.csv"
LOWLEVEL_CSV       = EXTRACT_DIR / "low_level.csv"
HIGHLEVEL_CSV      = EXTRACT_DIR / "high_level.csv"
LYRICS_JSONL       = EXTRACT_DIR / "lyrics.jsonl"
CLEAN_LYRICS_JSONL = EXTRACT_DIR / "clean_lyrics.jsonl"

# Failure / dead-letter files
MISSED_LOWLEVEL    = EXTRACT_DIR / "missed_low_level.csv"
MISSED_HIGHLEVEL   = EXTRACT_DIR / "missed_high_level.csv"
MISSED_LYRICS_CSV  = EXTRACT_DIR / "missed_lyrics.csv"
MISSED_CLEAN_JSONL = EXTRACT_DIR / "missed_clean_lyrics.jsonl"
DEAD_LETTER_JSONL  = EXTRACT_DIR / "dead_letter.jsonl"

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS = {".aac", ".flac", ".mp3", ".ogg", ".wav"}

SAMPLE_RATE_LOW = 22_050   # lowlevel DSP
SAMPLE_RATE_STD = 44_100   # highlevel / key extraction / loudness
SAMPLE_RATE_NN  = 16_000   # neural backbone input

# ---------------------------------------------------------------------------
# lowlevel DSP
# ---------------------------------------------------------------------------

FRAME_SIZE = 2048
HOP_SIZE   = 1024

# Explicit column schema for lowlevel.csv.
# Defined here so the CSV header is deterministic regardless of whether
# all frames contain spectral peaks (chroma may be absent for silent audio).
NOTE_NAMES = ["C", "Cs", "D", "Ds", "E", "F", "Fs", "G", "Gs", "A", "As", "B"]

LOWLEVEL_COLUMNS = (
    ["track_id"]
    + [f"mfcc_{i}_mean" for i in range(13)]
    + [f"mfcc_{i}_std"  for i in range(13)]
    + [
        "spectral_centroid_mean", "spectral_centroid_std",
        "spectral_rolloff_mean",  "spectral_rolloff_std",
        "spectral_flux_mean",     "spectral_flux_std",
        "zcr_mean", "zcr_std", "rms_mean",
        "bpm", "bpm_confidence",
    ]
    + [f"chroma_{n}_mean" for n in NOTE_NAMES]
    + [f"chroma_{n}_std"  for n in NOTE_NAMES]
)

# ---------------------------------------------------------------------------
# highlevel neural
# ---------------------------------------------------------------------------

# Hard cap to prevent TensorFlow OOM across subprocess workers.
MAX_HIGHLEVEL_WORKERS = 4

TOP_N_GENRES      = 5
TOP_N_MOOD_THEMES = 5
TOP_N_INSTRUMENTS = 8

# Model filenames relative to MODEL_DIR
EFFNET_BACKBONE = "discogs-effnet-bs64-1.pb"
TEMPOCNN_MODEL  = "deepsquare-k16-3.pb"

HIGHLEVEL_MODELS = {
    "genre_discogs400": "genre_discogs400-discogs-effnet-1.pb",
    "mood_aggressive":  "mood_aggressive-discogs-effnet-1.pb",
    "mood_happy":       "mood_happy-discogs-effnet-1.pb",
    "mood_party":       "mood_party-discogs-effnet-1.pb",
    "mood_relaxed":     "mood_relaxed-discogs-effnet-1.pb",
    "mood_sad":         "mood_sad-discogs-effnet-1.pb",
    "mood_acoustic":    "mood_acoustic-discogs-effnet-1.pb",
    "mood_electronic":  "mood_electronic-discogs-effnet-1.pb",
    "mood_theme":       "mtg_jamendo_moodtheme-discogs-effnet-1.pb",
    "danceability":     "danceability-discogs-effnet-1.pb",
    "approachability":  "approachability_regression-discogs-effnet-1.pb",
    "engagement":       "engagement_regression-discogs-effnet-1.pb",
    "tonal_atonal":     "tonal_atonal-discogs-effnet-1.pb",
    "voice_instrumental": "voice_instrumental-discogs-effnet-1.pb",
    "gender":           "gender-discogs-effnet-1.pb",
    "instrumentation":  "mtg_jamendo_instrument-discogs-effnet-1.pb",
    "timbre":           "timbre-discogs-effnet-1.pb",
}

# Models that use a non-default output layer
HIGHLEVEL_MODEL_OUTPUTS = {
    "approachability": "model/Identity",
    "engagement":      "model/Identity",
}

HIGHLEVEL_MODEL_INPUTS = {
    "genre_discogs400": "serving_default_model_Placeholder",
}

HIGHLEVEL_MODEL_OUTPUTS_OVERRIDE = {
    "genre_discogs400": "PartitionedCall:0",
}

# ---------------------------------------------------------------------------
# Lyrics fetching
# ---------------------------------------------------------------------------

LYRICS_DEFAULT_WORKERS  = 3
LYRICS_TARGET_RPS       = 10    # requests per second across all workers
LYRICS_MATCH_THRESHOLD  = 45    # min score to accept a Genius result
LYRICS_LOW_THRESHOLD    = 25    # relaxed threshold for interludes / skits
LYRICS_SHORT_TRACK_SEC  = 30    # skip tracks shorter than this
LYRICS_BATCH_SIZE       = 500   # checkpoint logging interval

LYRICS_DEFINITIVE_OUTCOMES = frozenset({
    "ok",
    "emoji_title",
    "short_track",
    "no_lyrics_on_genius",
    "missing_metadata",
})