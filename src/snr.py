# Estimate the signal-to-noise ratio (SNR) and the background-noise level of each clip, for the noise rule
# (score 0 if SNR < 8 dB and noise > -28 dB; step 1 of the notebook).
#
# SNR (dB) = loudness of the loudest parts (speech) - loudness of the quietest parts (background noise).
# A clip drowned in background noise has a low SNR.
# Also saves the noise level (dB) itself: a quiet but clear recording can have a low SNR too,
# so the noise rule uses both (low SNR and loud noise).
#
# Saves:  output_submission/snr_<train or test>.csv   (filename, label, snr_db, noise_db)
# Run:    .venv/Scripts/python src/snr.py         (training clips)
#         .venv/Scripts/python src/snr.py test    (test clips)

import os
import sys

import numpy as np
import pandas as pd
import soundfile as sf

# ---- Settings ----
# Which clips: "train" (default) or "test" (given after the script name)
SPLIT = "train"
if len(sys.argv) > 1:
    SPLIT = sys.argv[1]
CSV_FILE = f"data/{SPLIT}.csv"
AUDIO_FOLDER = f"data/{SPLIT}"
FRAME_SECONDS = 0.05   # cut the audio into 50 ms slices


def estimate_snr(path):
    """SNR (dB) and background-noise level (dB) of one clip."""
    audio, sample_rate = sf.read(path, dtype="float32")

    # Cut the audio into slices of 50 ms
    frame_length = int(sample_rate * FRAME_SECONDS)
    n_frames = len(audio) // frame_length
    frames = audio[: n_frames * frame_length].reshape(n_frames, frame_length)

    # Loudness of each slice, in decibels
    loudness = np.sqrt(np.mean(frames ** 2, axis=1))
    loudness = loudness[loudness > 0]          # skip slices of pure digital silence
    loudness_db = 20 * np.log10(loudness)

    speech_level = np.percentile(loudness_db, 95)  # loudest slices = speech
    noise_level = np.percentile(loudness_db, 10)   # quietest slices = background noise
    return speech_level - noise_level, noise_level


# ---- Compute SNR and noise level for every clip ----
clips = pd.read_csv(CSV_FILE)
snr_values = []
noise_values = []
for filename in clips["filename"]:
    snr, noise = estimate_snr(os.path.join(AUDIO_FOLDER, filename))
    snr_values.append(snr)
    noise_values.append(noise)
clips["snr_db"] = snr_values
clips["noise_db"] = noise_values

os.makedirs("output_submission", exist_ok=True)
clips.to_csv(f"output_submission/snr_{SPLIT}.csv", index=False)
print(f"Saved output_submission/snr_{SPLIT}.csv")
