"""
high_level.py - Neural feature extraction using Essentia TensorFlow models.

Extracts genre, mood, danceability, vocal/tonal properties, instrumentation,
rhythm, and audio embeddings (with temporal segmentation) from each track.

Single process mode is used because each TensorFlow model graph is expensive to
load, and multiprocessing can duplicate graph and runtime state.

Results are streamed directly to CSV as they complete to keep memory use flat,
which matters when each row contains multiple large embedding vectors.

Output: high_level.csv, missed_highlevel.csv
"""
# pyright: reportAttributeAccessIssue=false

import csv
import json
import os

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
# Model loading helpers
# ---------------------------------------------------------------------------

def _model_path(filename: str) -> str:
    return str(config.MODEL_DIR / filename)


def _model_exists(filename: str) -> bool:
    return os.path.exists(_model_path(filename))


def _load_json_labels(filename: str, key: str = "classes") -> list[str] | None:
    path = _model_path(filename)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f).get(key)
    except Exception:
        return None


def _infer_output_layer(pb_filename: str, fallback: str = "model/Softmax") -> str:
    """Read the companion JSON metadata to determine the output node name."""
    json_path = _model_path(pb_filename.replace(".pb", ".json"))
    if not os.path.exists(json_path):
        return fallback
    try:
        with open(json_path, encoding="utf-8") as f:
            outputs = json.load(f).get("schema", {}).get("outputs", [])
        if outputs:
            return outputs[0]["name"]
    except Exception:
        pass
    return fallback


def _build_effnet_backbone():
    f = config.EFFNET_BACKBONE
    if not _model_exists(f):
        return None
    return es.TensorflowPredictEffnetDiscogs(
        graphFilename=_model_path(f), output="PartitionedCall:1"
    )


def _build_2d_model(name: str):
    filename = config.HIGHLEVEL_MODELS.get(name)
    if not filename or not _model_exists(filename):
        return None
    output = (
        config.HIGHLEVEL_MODEL_OUTPUTS_OVERRIDE.get(name)
        or config.HIGHLEVEL_MODEL_OUTPUTS.get(name)
        or _infer_output_layer(filename)
    )
    kwargs: dict = {"graphFilename": _model_path(filename), "output": output}
    if name in config.HIGHLEVEL_MODEL_INPUTS:
        kwargs["input"] = config.HIGHLEVEL_MODEL_INPUTS[name]
    return es.TensorflowPredict2D(**kwargs)


def _build_tempocnn():
    f = config.TEMPOCNN_MODEL
    if not _model_exists(f):
        return None
    return es.TempoCNN(graphFilename=_model_path(f))


# ---------------------------------------------------------------------------
# Per-process model state
# ---------------------------------------------------------------------------

_models: dict = {}
_tempocnn = None


def _init_models_once() -> None:
    """
    Load all TF models once in the main process.
    """
    global _models, _tempocnn

    if _models:
        return

    _models["backbone"] = _build_effnet_backbone()

    for name in config.HIGHLEVEL_MODELS:
        _models[name] = _build_2d_model(name)

    _models["genre_discogs400_labels"] = _load_json_labels(
        config.EFFNET_BACKBONE.replace(".pb", ".json")
    )
    _models["mood_theme_labels"] = _load_json_labels(
        config.HIGHLEVEL_MODELS["mood_theme"].replace(".pb", ".json")
    )
    _models["instrumentation_labels"] = _load_json_labels(
        config.HIGHLEVEL_MODELS["instrumentation"].replace(".pb", ".json")
    )

    _tempocnn = _build_tempocnn()


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------

def _mean(arr: np.ndarray) -> np.ndarray:
    return np.mean(arr, axis=0)


def _std(arr: np.ndarray) -> np.ndarray:
    return np.std(arr, axis=0)


def _safe_predict(model, embeddings: np.ndarray, name: str) -> np.ndarray | None:
    try:
        return model(embeddings)
    except Exception as e:
        log.warning("Prediction failed [%s]: %s", name, e)
        return None


def _top_n_labels(probs: np.ndarray, labels: list[str], n: int, prefix: str) -> dict:
    top_idx = np.argsort(probs)[::-1][:n]
    out = {}
    for rank, idx in enumerate(top_idx, 1):
        label = labels[idx] if idx < len(labels) else f"class_{idx}"
        out[f"{prefix}_top{rank}"]      = label
        out[f"{prefix}_top{rank}_prob"] = round(float(probs[idx]), 4)
    return out


def _binary_prob(probs: np.ndarray, positive_index: int = 1) -> float:
    return round(float(probs[positive_index]), 4)


# ---------------------------------------------------------------------------
# Feature extraction sub-routines
# ---------------------------------------------------------------------------

def _extract_genre(embeddings: np.ndarray) -> dict:
    model  = _models.get("genre_discogs400")
    labels = _models.get("genre_discogs400_labels")
    if model is None or labels is None:
        return {}
    raw = _safe_predict(model, embeddings, "genre_discogs400")
    if raw is None:
        return {}
    return _top_n_labels(_mean(raw), labels, config.TOP_N_GENRES, "genre")


def _extract_mood(embeddings: np.ndarray) -> dict:
    feats = {}

    for tag in ("aggressive", "happy", "party", "relaxed", "sad"):
        model = _models.get(f"mood_{tag}")
        if model is None:
            continue
        raw = _safe_predict(model, embeddings, f"mood_{tag}")
        if raw is not None:
            pooled = raw[:, 1]
            feats[f"mood_{tag}"]     = round(float(np.mean(pooled)), 4)
            feats[f"mood_{tag}_std"] = round(float(np.std(pooled)),  4)

    for tag in ("acoustic", "electronic"):
        model = _models.get(f"mood_{tag}")
        if model is None:
            continue
        raw = _safe_predict(model, embeddings, f"mood_{tag}")
        if raw is not None:
            feats[f"{tag}_prob"] = _binary_prob(_mean(raw))

    theme_model  = _models.get("mood_theme")
    theme_labels = _models.get("mood_theme_labels")
    if theme_model is not None and theme_labels is not None:
        raw = _safe_predict(theme_model, embeddings, "mood_theme")
        if raw is not None:
            feats.update(_top_n_labels(_mean(raw), theme_labels, config.TOP_N_MOOD_THEMES, "theme"))

    return feats


def _extract_dance_and_affect(embeddings: np.ndarray, mood_feats: dict) -> dict:
    feats = {}

    dance_model = _models.get("danceability")
    if dance_model is not None:
        raw = _safe_predict(dance_model, embeddings, "danceability")
        if raw is not None:
            pooled = raw[:, 1]
            feats["danceability"]     = round(float(np.mean(pooled)), 4)
            feats["danceability_std"] = round(float(np.std(pooled)),  4)

    for key in ("approachability", "engagement"):
        model = _models.get(key)
        if model is None:
            continue
        raw = _safe_predict(model, embeddings, key)
        if raw is not None:
            feats[f"{key}_mean"] = round(float(np.mean(raw)), 4)
            feats[f"{key}_std"]  = round(float(np.std(raw)),  4)

    # Proxy valence/arousal derived from mood scores (Russell circumplex approximation)
    happy = mood_feats.get("mood_happy",      0.5)
    sad   = mood_feats.get("mood_sad",        0.5)
    aggr  = mood_feats.get("mood_aggressive", 0.5)
    relax = mood_feats.get("mood_relaxed",    0.5)
    party = mood_feats.get("mood_party",      0.5)

    feats["valence_mean_proxy"] = round(float(((happy + relax) - (sad + aggr) + 2.0) / 4.0), 4)
    feats["arousal_mean_proxy"] = round(float(((aggr + party)  - (sad + relax) + 2.0) / 4.0), 4)

    return feats


def _extract_tonal_and_voice(embeddings: np.ndarray) -> dict:
    feats = {}

    for name, out_key in [("tonal_atonal", "tonal_prob"), ("voice_instrumental", "voice_prob")]:
        model = _models.get(name)
        if model is not None:
            raw = _safe_predict(model, embeddings, name)
            if raw is not None:
                feats[out_key] = _binary_prob(_mean(raw))

    gender_model = _models.get("gender")
    if gender_model is not None:
        raw = _safe_predict(gender_model, embeddings, "gender")
        if raw is not None:
            probs = _mean(raw)
            feats["gender_female_prob"] = round(float(probs[0]), 4)
            feats["gender_male_prob"]   = round(float(probs[1]), 4)

    instr_model  = _models.get("instrumentation")
    instr_labels = _models.get("instrumentation_labels")
    if instr_model is not None and instr_labels is not None:
        raw = _safe_predict(instr_model, embeddings, "instrumentation")
        if raw is not None:
            feats.update(_top_n_labels(_mean(raw), instr_labels, config.TOP_N_INSTRUMENTS, "instrument"))

    timbre_model = _models.get("timbre")
    if timbre_model is not None:
        raw = _safe_predict(timbre_model, embeddings, "timbre")
        if raw is not None:
            probs = _mean(raw)
            feats["timbre_bright_prob"] = round(float(probs[0]), 4)
            feats["timbre_dark_prob"]   = round(float(probs[1]), 4)

    return feats


def _extract_tonality(audio_44k: np.ndarray) -> dict:
    feats = {}
    try:
        key, scale, strength = es.KeyExtractor()(audio_44k)
        feats["key"]          = key
        feats["scale"]        = scale
        feats["key_strength"] = round(float(strength), 4)
    except Exception as e:
        log.debug("KeyExtractor failed: %s", e)

    try:
        stereo = np.column_stack([audio_44k, audio_44k])
        _, _, integrated, loudness_range = es.LoudnessEBUR128(sampleRate=44100)(stereo)
        feats["lufs_integrated"] = round(float(integrated),    4)
        feats["loudness_range"]  = round(float(loudness_range), 4)
    except Exception as e:
        log.debug("LoudnessEBUR128 failed: %s", e)

    return feats


def _extract_rhythm(audio_16k: np.ndarray, audio_44k: np.ndarray) -> dict:
    feats = {}

    if _tempocnn is not None:
        try:
            out = _tempocnn(audio_16k)
            bpm = float(out.item() if isinstance(out, np.ndarray) and out.size == 1 else out[0])
            feats["bpm"] = round(bpm, 2)
        except Exception as e:
            log.debug("TempoCNN failed: %s", e)

    try:
        bpm_r, beats, confidence, _, intervals = es.RhythmExtractor2013(method="multifeature")(audio_44k)
        if "bpm" not in feats:
            feats["bpm"] = round(float(bpm_r), 2)
        feats["beats_confidence"]    = round(float(confidence), 4)
        feats["beats_count"]         = int(len(beats))
        if len(intervals) > 0:
            feats["beat_interval_mean"] = round(float(np.mean(intervals)), 4)
            feats["beat_interval_std"]  = round(float(np.std(intervals)),  4)
    except Exception:
        pass

    try:
        feats["onset_rate"] = round(float(es.OnsetRate()(audio_44k)[1]), 4)
    except Exception:
        pass

    return feats


# ---------------------------------------------------------------------------
# Per-track worker function
# ---------------------------------------------------------------------------

def _process_track(track_id: str, file_path: str) -> dict:
    if not os.path.exists(file_path):
        return {"track_id": track_id, "file_path": file_path, "status": "missing_file"}

    try:
        audio_44k = es.MonoLoader(filename=file_path, sampleRate=config.SAMPLE_RATE_STD)()
        audio_16k = es.Resample(
            inputSampleRate=config.SAMPLE_RATE_STD,
            outputSampleRate=config.SAMPLE_RATE_NN,
        )(audio_44k)
    except Exception:
        return {"track_id": track_id, "file_path": file_path, "status": "load_failed"}

    feats: dict = {}
    backbone = _models.get("backbone")

    if backbone is not None:
        try:
            embeddings = backbone(audio_16k)
        except Exception:
            return {"track_id": track_id, "file_path": file_path, "status": "backbone_failed"}

        n = len(embeddings)

        # Global pooled embedding
        feats["audio_embedding"] = ",".join(map(str, np.round(_mean(embeddings), 6)))

        # Temporal segments: start / mid / end third
        if n >= 3:
            t = n // 3
            feats["audio_embedding_start"] = ",".join(map(str, np.round(np.mean(embeddings[:t],       axis=0), 6)))
            feats["audio_embedding_mid"]   = ",".join(map(str, np.round(np.mean(embeddings[t:2*t],    axis=0), 6)))
            feats["audio_embedding_end"]   = ",".join(map(str, np.round(np.mean(embeddings[2*t:],     axis=0), 6)))
        else:
            feats["audio_embedding_start"] = feats["audio_embedding_mid"] = feats["audio_embedding_end"] = feats["audio_embedding"]

        feats["embedding_macro_variance"] = round(float(np.mean(_std(embeddings))), 4)

        feats.update(_extract_genre(embeddings))
        mood_feats = _extract_mood(embeddings)
        feats.update(mood_feats)
        feats.update(_extract_dance_and_affect(embeddings, mood_feats))
        feats.update(_extract_tonal_and_voice(embeddings))

    feats.update(_extract_tonality(audio_44k))
    feats.update(_extract_rhythm(audio_16k, audio_44k))

    feats["track_id"] = track_id
    feats["status"]   = "success"
    return feats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    config.EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    _init_models_once()

    metadata_df = pd.read_csv(config.METADATA_CSV)
    log.info("Loaded %d tracks from metadata", len(metadata_df))
    log.info("Starting neural extraction")

    # We don't know the column list until the first result arrives (depends on
    # which models are available), so we initialise the CSV writer lazily.
    csv_columns: list[str] | None = None
    missed: list[dict] = []
    success = 0

    with open(config.HIGHLEVEL_CSV, "w", newline="") as f:
        writer = None

        for row in tqdm(
            metadata_df.itertuples(index=False),
            total=len(metadata_df),
            desc="Neural extraction",
            dynamic_ncols=True,
        ):
            result = _process_track(str(row.track_id), str(row.file_path))
            status = result.pop("status")

            if status == "success":
                if writer is None:
                    # Establish schema from first successful result
                    csv_columns = ["track_id"] + [c for c in result if c != "track_id"]
                    writer = csv.DictWriter(
                        f,
                        fieldnames=csv_columns,
                        extrasaction="ignore",
                    )
                    writer.writeheader()

                writer.writerow(result)
                success += 1
            else:
                missed.append({
                    "track_id":  result["track_id"],
                    "file_path": result.get("file_path", ""),
                    "reason":    status,
                })

    log.info("Done - success: %d, failed: %d", success, len(missed))

    if missed:
        pd.DataFrame(missed).to_csv(config.MISSED_HIGHLEVEL, index=False)
        log.warning("Failed tracks written to %s", config.MISSED_HIGHLEVEL)


if __name__ == "__main__":
    main()