# Find which training clips come from the same speaker, so cross-validation can keep each speaker in one fold.
#
# Why: many speakers recorded several answers (consecutive file numbers, different questions, similar scores).
#      With random folds, another clip of the same speaker is often in the training part, and the model can
#      recognise the voice instead of judging the grammar. The test clips come from new speakers, so random-fold
#      CV looks better than it should - most of all for audio models.
# How: a speaker-recognition model (microsoft/wavlm-base-plus-sv) turns every clip into a voice fingerprint
#      (an "x-vector"). The average fingerprint is subtracted, and fingerprints are compared by cosine similarity.
#      Clips are grouped bottom-up: two groups are joined while their average similarity is at least SAME_SPEAKER.
#      (Random pairs of clips go above 0.89 only about 1 time in 1000; clips of one speaker are usually above 0.9.)
# Only clips scored 1-5 are grouped (the clips used for training and CV).
#
# Saves:  output_submission/speaker_groups.csv   (filename, speaker)
# Run:    .venv/Scripts/python src/speakers.py

import os

import truststore
truststore.inject_into_ssl()  # use the Windows certificate store, otherwise the model download fails

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from transformers import AutoFeatureExtractor, WavLMForXVector

# ---- Settings ----
MODEL_ID = "microsoft/wavlm-base-plus-sv"
OUTPUT_CSV = "output_submission/speaker_groups.csv"
SAME_SPEAKER = 0.85        # minimum average similarity for two groups of clips to count as one speaker


def fingerprints(paths, feature_extractor, model):
    """One voice fingerprint (x-vector) per clip, from the whole clip."""
    result = []
    with torch.no_grad():
        for path in tqdm(paths):
            audio, sample_rate = sf.read(path, dtype="float32")
            inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt").to("cuda")
            result.append(model(**inputs).embeddings[0].cpu().numpy())
    return np.array(result)


def centre_and_normalise(x, average):
    """Subtract the average fingerprint and scale to length 1, so a dot product is the cosine similarity."""
    x = x - average
    return x / np.linalg.norm(x, axis=1, keepdims=True)


# ---- Fingerprints ----
clips = pd.read_csv("output_submission/transcripts_train.csv")
clips = clips[clips["label"] > 0].reset_index(drop=True)

feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
model = WavLMForXVector.from_pretrained(MODEL_ID, dtype=torch.float32).to("cuda").eval()
train_x = fingerprints([os.path.join("data/train", f) for f in clips["filename"]], feature_extractor, model)

average = train_x.mean(axis=0)
train_x = centre_and_normalise(train_x, average)


# ---- Group the training clips by speaker ----
grouping = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average",
                                   distance_threshold=1 - SAME_SPEAKER)   # distance = 1 - similarity
clips["speaker"] = grouping.fit_predict(train_x)
clips[["filename", "speaker"]].to_csv(OUTPUT_CSV, index=False)

sizes = clips["speaker"].value_counts()
print(f"\n{len(clips)} clips -> {len(sizes)} speakers; clips per speaker: largest {sizes.max()}, "
      f"speakers with one clip {(sizes == 1).sum()}")

print("Saved", OUTPUT_CSV)
