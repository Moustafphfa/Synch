"""
low_level.py — DSP feature extraction (MFCC, chroma, spectral, rhythm).

Uses Essentia with a multiprocessing pool. Each worker initialises its own
algorithm instances to avoid segfaults from shared Essentia state across
processes.

Results are written to CSV incrementally (one row per completed track) to
keep memory use flat regardless of catalog size.

Output: low_level.csv, missed_lowlevel.csv
"""
# pyright: reportAttributeAccessIssue=false

import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import essentia
import essentia.standard as es
essentia.log.warningActive = False
essentia.log.infoActive    = False

import numpy as np
import pandas as pd
from tqdm import tqdm

import config
from utils.logging_setup import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Per-worker algorithm state (initialised once per subprocess)
# ---------------------------------------------------------------------------

_algs: dict = {}


def _init_worker() -> None:
    """
    Instantiate Essentia algorithms once per worker process.

    Creating these inside each call would add significant overhead; sharing
    them across calls within a process is safe because each process has its
    own memory space.
    """
    global _algs
    sr = config.SAMPLE_RATE_LOW
    _algs.update({
        "window":         es.Windowing(type="hann"),
        "spectrum":       es.Spectrum(),
        "mfcc":           es.MFCC(sampleRate=sr),
        "centroid":       es.Centroid(range=sr / 2),
        "rolloff":        es.RollOff(sampleRate=sr),
        "spectral_peaks": es.SpectralPeaks(sampleRate=sr),
        "hpcp":           es.HPCP(sampleRate=sr, size=12),
        "rms":            es.RMS(),
        "flux":           es.Flux(),
        "zcr":            es.ZeroCrossingRate(),
        "rhythm":         es.RhythmExtractor2013(method="multifeature"),
    })


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _extract_features(track_id: str, file_path: str) -> dict:
    """
    Compute DSP features for a single audio file.

    Returns a dict with keys matching config.LOWLEVEL_COLUMNS plus a
    'status' key ('success' or a short error label).
    """
    if not os.path.exists(file_path):
        return {"track_id": track_id, "file_path": file_path, "status": "missing_file"}

    try:
        audio = es.MonoLoader(filename=file_path, sampleRate=config.SAMPLE_RATE_LOW)()
    except Exception as e:
        logging.debug("Audio load failed for %s: %s", track_id, e)
        return {"track_id": track_id, "file_path": file_path, "status": "loader_failed"}

    mfccs, centroids, rolloffs, hpcps, zcrs, fluxes = [], [], [], [], [], []

    try:
        for frame in es.FrameGenerator(
            audio,
            frameSize=config.FRAME_SIZE,
            hopSize=config.HOP_SIZE,
            startFromZero=True,
        ):
            zcrs.append(_algs["zcr"](frame))
            spec = _algs["spectrum"](_algs["window"](frame))
            _, mfcc_bands = _algs["mfcc"](spec)
            mfccs.append(mfcc_bands)
            centroids.append(_algs["centroid"](spec))
            rolloffs.append(_algs["rolloff"](spec))
            fluxes.append(_algs["flux"](spec))

            freqs, mags = _algs["spectral_peaks"](spec)
            if len(freqs) > 0:
                hpcps.append(_algs["hpcp"](freqs, mags))

        bpm, _, bpm_confidence, _, _ = _algs["rhythm"](audio)

    except Exception as e:
        logging.error("Feature extraction failed for %s: %s", track_id, e)
        return {"track_id": track_id, "file_path": file_path, "status": "extraction_failed"}

    if not mfccs:
        return {"track_id": track_id, "file_path": file_path, "status": "empty_signal"}

    mfcc_arr  = np.array(mfccs)
    hpcp_arr  = np.array(hpcps) if hpcps else np.empty((0, 12))
    note_names = config.NOTE_NAMES

    feats: dict = {"track_id": track_id}

    for i in range(mfcc_arr.shape[1]):
        feats[f"mfcc_{i}_mean"] = float(np.mean(mfcc_arr[:, i]))
        feats[f"mfcc_{i}_std"]  = float(np.std(mfcc_arr[:, i]))

    feats.update({
        "spectral_centroid_mean": float(np.mean(centroids)) if centroids else 0.0,
        "spectral_centroid_std":  float(np.std(centroids))  if centroids else 0.0,
        "spectral_rolloff_mean":  float(np.mean(rolloffs))  if rolloffs  else 0.0,
        "spectral_rolloff_std":   float(np.std(rolloffs))   if rolloffs  else 0.0,
        "spectral_flux_mean":     float(np.mean(fluxes))    if fluxes    else 0.0,
        "spectral_flux_std":      float(np.std(fluxes))     if fluxes    else 0.0,
        "zcr_mean":               float(np.mean(zcrs))      if zcrs      else 0.0,
        "zcr_std":                float(np.std(zcrs))       if zcrs      else 0.0,
        "rms_mean":               float(_algs["rms"](audio)),
        "bpm":                    float(bpm),
        "bpm_confidence":         float(bpm_confidence),
    })

    for i, name in enumerate(note_names):
        if hpcp_arr.shape[0] > 0:
            feats[f"chroma_{name}_mean"] = float(np.mean(hpcp_arr[:, i]))
            feats[f"chroma_{name}_std"]  = float(np.std(hpcp_arr[:, i]))
        else:
            feats[f"chroma_{name}_mean"] = 0.0
            feats[f"chroma_{name}_std"]  = 0.0

    feats["status"] = "success"
    return feats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config.EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(config.METADATA_CSV)
    log.info("Loaded %d tracks from metadata", len(df))

    # Write CSV header upfront so the file is valid even if all workers fail.
    with open(config.LOWLEVEL_CSV, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=config.LOWLEVEL_COLUMNS).writeheader()

    missed: list[dict] = []
    n_workers = os.cpu_count() or 1
    log.info("Starting DSP pool with %d workers", n_workers)

    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as pool:
        futures = {
            pool.submit(_extract_features, str(row.track_id), str(row.file_path)): row.track_id
            for row in df.itertuples(index=False)
        }

        with open(config.LOWLEVEL_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=config.LOWLEVEL_COLUMNS)
            
            for n, future in enumerate(tqdm(as_completed(futures), total=len(futures),
                               desc="DSP extraction", dynamic_ncols=True)):
                result = future.result()
                status = result.pop("status")

                if status == "success":
                    writer.writerow({col: result.get(col, "") for col in config.LOWLEVEL_COLUMNS})
                    
                    # Flush to disk every 100 tracks for crash resilience
                    if n % 100 == 0:
                        f.flush()
                else:
                    missed.append({
                        "track_id":  result["track_id"],
                        "file_path": result.get("file_path", ""),
                        "reason":    status,
                    })

    log.info("Done — success: %d, failed: %d", len(df) - len(missed), len(missed))

    if missed:
        pd.DataFrame(missed).to_csv(config.MISSED_LOWLEVEL, index=False)
        log.warning("Failed tracks written to %s", config.MISSED_LOWLEVEL)


if __name__ == "__main__":
    main()