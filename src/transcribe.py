# Transcribe each clip word for word with CrisperWhisper 2.0.
#
# Clips longer than 30 s are cut at pauses into pieces of 30 s or less. Each piece is transcribed
# on its own and the texts are joined. (CrisperWhisper's own long-audio mode got stuck repeating
# phrases in ~17% of clips; cutting at pauses brought this down to 6 of 769.)
#
# If a piece's text still loops (compression ratio > 2.4), that piece is cut again at its quietest
# 200 ms in the middle third and the halves are transcribed on their own; a half that still loops
# is cut again the same way (down to 2 s). This fixed the remaining 6 clips.
# Compression ratio = length of the text / length of the zipped text. Repeated text zips very well,
# so a loop gives a high ratio (normal pieces 0.3-1.9, looping pieces 6-18).
#
# For every clip it saves:  output_submission/transcripts/<train or test>/<clip name>.txt
# At the end it saves:      output_submission/transcripts_<train or test>.csv   (filename, label, transcript)
#
# Every clip is transcribed again on each run (existing .txt files are overwritten).
# Run:  .venv/Scripts/python src/transcribe.py         (training clips)
#       .venv/Scripts/python src/transcribe.py test    (test clips)

import os
import sys
import zlib

import truststore
truststore.inject_into_ssl()  # use the Windows certificate store, otherwise the model download fails

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from crisperwhisper import CrisperWhisperModel
from tqdm import tqdm

# ---- Settings ----
# Which clips: "train" (default) or "test" (given after the script name)
SPLIT = "train"
if len(sys.argv) > 1:
    SPLIT = sys.argv[1]
CSV_FILE = f"data/{SPLIT}.csv"
AUDIO_FOLDER = f"data/{SPLIT}"
OUTPUT_FOLDER = f"output_submission/transcripts/{SPLIT}"
OUTPUT_CSV = f"output_submission/transcripts_{SPLIT}.csv"

FRAME_SECONDS = 0.05       # loudness is measured on 50 ms slices (as in snr.py)
MAX_PIECE_SECONDS = 30     # the model hears at most 30 s at a time
SEARCH_FROM_SECONDS = 20   # look for a pause between 20 s ...
SEARCH_TO_SECONDS = 28     # ... and 28 s after the previous cut
LOOP_RATIO = 2.4           # compression ratio above this = the piece looped (Whisper's own threshold)
MIN_RECUT_SECONDS = 2      # don't cut parts shorter than this again


def stretch_loudness(audio, sample_rate):
    """Loudness (dB) of each 200 ms stretch, one value every 50 ms."""
    # Loudness of each 50 ms slice, in decibels
    frame_length = int(sample_rate * FRAME_SECONDS)
    n_frames = len(audio) // frame_length
    frames = audio[: n_frames * frame_length].reshape(n_frames, frame_length)
    loudness_db = 20 * np.log10(np.sqrt(np.mean(frames ** 2, axis=1)) + 1e-10)  # + 1e-10 avoids log(0)

    # Loudness of each 200 ms stretch = average of 4 slices in a row
    # (so one quiet slice in the middle of a word can't be chosen as a pause)
    return np.convolve(loudness_db, np.ones(4) / 4, mode="valid")


def find_cuts(audio, sample_rate):
    """Return the cut times (in seconds) so that every piece is at most 30 s long."""
    stretch_db = stretch_loudness(audio, sample_rate)
    cuts = []
    start = 0.0
    total_seconds = len(audio) / sample_rate
    while total_seconds - start > MAX_PIECE_SECONDS:
        first = int((start + SEARCH_FROM_SECONDS) / FRAME_SECONDS)
        last = int((start + SEARCH_TO_SECONDS) / FRAME_SECONDS)
        quietest = first + np.argmin(stretch_db[first:last])  # start of the quietest 200 ms stretch
        cut = (quietest + 2) * FRAME_SECONDS                  # cut in the middle of that stretch
        cuts.append(cut)
        start = cut
    return cuts


def middle_cut(piece, sample_rate):
    """Return the time (in seconds) of the quietest 200 ms in the middle third of a piece."""
    stretch_db = stretch_loudness(piece, sample_rate)
    first = len(stretch_db) // 3
    last = 2 * len(stretch_db) // 3
    quietest = first + np.argmin(stretch_db[first:last])
    return (quietest + 2) * FRAME_SECONDS


def compression_ratio(text):
    """Length of the text / length of the zipped text: high (> 2.4) when the text repeats itself in a loop."""
    data = text.encode("utf-8")
    if len(data) == 0:
        return 0.0
    return len(data) / len(zlib.compress(data))


def transcribe_piece(piece, sample_rate):
    """Text of one piece of audio (at most 30 s)."""
    result = model.transcribe(piece, sr=sample_rate, language="en")  # piece <= 30 s: one window
    return result.text.strip()


def transcribe_without_loops(piece, sample_rate, piece_start, filename):
    """Transcribe a piece. If the text loops, cut that part in its middle and transcribe the two
    halves alone; repeat until no part loops (or a part is shorter than MIN_RECUT_SECONDS)."""
    parts = [(piece, piece_start)]   # parts still to transcribe, in time order: (audio, start time)
    texts = []
    while len(parts) > 0:
        part, part_start = parts.pop(0)          # take the first part from the list
        text = transcribe_piece(part, sample_rate)
        ratio = compression_ratio(text)
        part_seconds = len(part) / sample_rate

        if ratio > LOOP_RATIO and part_seconds >= MIN_RECUT_SECONDS:
            cut = middle_cut(part, sample_rate)
            tqdm.write(f"{filename}: {part_start:.1f}-{part_start + part_seconds:.1f} s looped"
                       f" (ratio {ratio:.2f}) -> re-cut at {part_start + cut:.1f} s")
            first_half = (part[: int(cut * sample_rate)], part_start)
            second_half = (part[int(cut * sample_rate) :], part_start + cut)
            parts = [first_half, second_half] + parts   # put both halves at the front, in order
        else:
            if ratio > LOOP_RATIO:
                tqdm.write(f"WARNING {filename}: {part_start:.1f}-{part_start + part_seconds:.1f} s"
                           f" still loops but is too short to cut again")
            texts.append(text)

    return " ".join(texts).strip()


def transcribe_with_pause_cuts(path):
    """Cut the clip at pauses, transcribe each piece alone (re-cutting any piece that loops), join the texts."""
    audio, sample_rate = sf.read(path, dtype="float32")
    cuts = find_cuts(audio, sample_rate)

    boundaries = [0.0] + cuts + [len(audio) / sample_rate]
    texts = []
    for i in range(len(boundaries) - 1):
        piece = audio[int(boundaries[i] * sample_rate) : int(boundaries[i + 1] * sample_rate)]
        texts.append(transcribe_without_loops(piece, sample_rate, boundaries[i], os.path.basename(path)))
    return " ".join(texts)


# ---- Load the model ----
# "large" = nyralabs/CrisperWhisper2.0_large
# backend="transformers" = run with PyTorch (the faster "ct2" backend is Linux only)
# compute_type="float32" = full precision: slower than float16, but no rounding effects on the words
print("Loading model...")
model = CrisperWhisperModel("large", backend="transformers", device="cuda", compute_type="float32")


# ---- Transcribe each clip ----
clips = pd.read_csv(CSV_FILE)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for filename in tqdm(clips["filename"]):
    txt_path = os.path.join(OUTPUT_FOLDER, filename.replace(".wav", ".txt"))
    text = transcribe_with_pause_cuts(os.path.join(AUDIO_FOLDER, filename))

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(text)

    # Give GPU memory that PyTorch kept for reuse back to the GPU, so it can't pile up over a long run
    # and spill into (slow) shared memory
    torch.cuda.empty_cache()


# ---- Put all transcripts into one CSV ----
transcripts = []
for filename in clips["filename"]:
    txt_path = os.path.join(OUTPUT_FOLDER, filename.replace(".wav", ".txt"))
    with open(txt_path, encoding="utf-8") as f:
        transcripts.append(f.read())

clips["transcript"] = transcripts
clips.to_csv(OUTPUT_CSV, index=False)
print("Saved", OUTPUT_CSV)
