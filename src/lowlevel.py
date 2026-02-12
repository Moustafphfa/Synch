import os
import logging
import essentia.standard as es
import pandas as pd
import numpy as np


OUTPUT_DIR = "Path"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "lowlevel.csv")
METADATA_CSV = os.path.join(OUTPUT_DIR, "metadata.csv")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def extract_lowlevel(file_path):
    # Computes MFCC and coarse spectral statistics per track.
    try:
        audio = es.MonoLoader(filename=file_path)()
    except Exception as e:
        logging.error("Failed loading %s: %s", file_path, e)
        return None

    frame_size = 2048
    hop_size = 1024
    window = es.Windowing(type='hann')
    spectrum = es.Spectrum()
    mfcc = es.MFCC()

    mfcc_coeffs, spectral_centroids, spectral_flatnesses, spectral_fluxes = [], [], [], []
    w_prev = np.zeros(frame_size // 2 + 1)

    for frame in es.FrameGenerator(audio, frameSize=frame_size, hopSize=hop_size, startFromZero=True):
        w = window(frame)
        spec = spectrum(w)
        _, mfcc_c = mfcc(spec)
        mfcc_coeffs.append(mfcc_c)
        spectral_centroids.append(es.Centroid()(spec))
        spectral_flatnesses.append(es.Flatness()(spec))
        spectral_fluxes.append(np.sum((spec - w_prev) ** 2))
        w_prev = spec

    mfcc_coeffs = np.array(mfcc_coeffs)
    spectral_centroids = np.array(spectral_centroids)
    spectral_flatnesses = np.array(spectral_flatnesses)
    spectral_fluxes = np.array(spectral_fluxes)
    rms_mean = es.RMS()(audio)

    return {
        'mfcc_1_mean': np.mean(mfcc_coeffs[:,0]) if mfcc_coeffs.size else 0,
        'mfcc_1_std': np.std(mfcc_coeffs[:,0]) if mfcc_coeffs.size else 0,
        'mfcc_13_mean': np.mean(mfcc_coeffs[:,12]) if mfcc_coeffs.shape[1] > 12 else 0,
        'spectral_centroid_mean': np.mean(spectral_centroids) if spectral_centroids.size else 0,
        'spectral_centroid_std': np.std(spectral_centroids) if spectral_centroids.size else 0,
        'spectral_flatness_mean': np.mean(spectral_flatnesses) if spectral_flatnesses.size else 0,
        'spectral_flux_mean': np.mean(spectral_fluxes) if spectral_fluxes.size else 0,
        'rms_mean': rms_mean
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load track metadata
    metadata_df = pd.read_csv(METADATA_CSV)
    logging.info("Loaded metadata: %d tracks", len(metadata_df))

    all_features = []
    processed = 0

    for _, row in metadata_df.iterrows():
        file_path = row['file_path']
        track_id = row['track_id']

        if not os.path.exists(file_path):
            logging.warning("File not found: %s", file_path)
            continue

        logging.info("Processing: %s", file_path)
        feats = extract_lowlevel(file_path)

        if feats:
            feats['track_id'] = track_id
            all_features.append(feats)
            processed += 1

    if all_features:
        df = pd.DataFrame(all_features)
        df = df[['track_id'] + [c for c in df.columns if c != 'track_id']]
        df.to_csv(OUTPUT_CSV, index=False)
        logging.info("Done: %d tracks → %s", processed, OUTPUT_CSV)
    else:
        logging.warning("No features extracted!")


if __name__ == "__main__":
    main()