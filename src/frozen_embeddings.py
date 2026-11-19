# Frozen embeddings of the five frozen parts of the blend, one vector per clip (step 4 of the notebook).
#
# Each part: a pretrained model, not trained here. The clip (or its transcript) goes through the model; the output of
# one chosen layer (best layer in speaker-grouped CV) gives one vector per frame (audio) or token (text); the average
# over all frames/tokens is the clip's vector. The notebook fits Ridge on these vectors (step 6).
#   whisper_l32           CrisperWhisper 2.0 large encoder, layer 32 of 32. Whisper reads at most 30 s: the clip is cut
#                         into 30 s windows (a last piece under 0.5 s is skipped), the padding of the last window is
#                         left out of the average (50 frames per second of real audio).
#   phi4_audio_l23        Phi-4-multimodal's audio encoder, layer 23 of 24. Only the audio encoder is loaded (all its
#                         weights are in the first of the 3 weight files; their names are renamed as in the official
#                         conversion script). The clip is cut into equal pieces of at most 40 s (the encoder's window),
#                         so there is no padding.
#   parakeet_ctc_l36      Parakeet-CTC-1.1B encoder, layer 36 of 42, whole clip at once.
#   parakeet_tdt_l20      Parakeet-TDT-0.6B-v3 encoder, layer 20 of 24, whole clip at once.
#   deberta_v3_large_l21  DeBERTa-v3-large, layer 21 of 24, on the transcript (sound tags like [breath] removed).
# The sixth input, the GED error embedding, is already saved by src/error_embedding.py.
#
# Saves:  output_submission/embeddings/<part>_train.npy   (training clips scored 1-5, order of transcripts_train.csv)
#         output_submission/embeddings/<part>_test.npy    (test clips, order of data/test.csv)
#         Every part is computed again on each run (existing files are overwritten).
# Run:    .venv/Scripts/python src/frozen_embeddings.py   (all parts, about 17 min on an RTX 5080 laptop GPU)

import os
import re

import truststore
truststore.inject_into_ssl()  # use the Windows certificate store, otherwise the model download fails

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from tqdm import tqdm
from transformers import (AutoFeatureExtractor, AutoModel, AutoProcessor, AutoTokenizer, ParakeetForCTC,
                          ParakeetForTDT, Phi4MultimodalAudioConfig, Phi4MultimodalAudioModel,
                          Phi4MultimodalFeatureExtractor, WhisperModel)

EMBEDDING_FOLDER = "output_submission/embeddings"


def clean_text(text):
    """Remove sound tags like [breath] (fillers [UH] and [UM] are kept), as in the other scripts."""
    return " ".join(re.sub(r"\[[a-z]+\]", " ", text).split())


def embed_whisper(paths):
    """Whisper encoder, layer 32: average over the real-audio frames of all 30 s windows of each clip."""
    model_id = "nyralabs/CrisperWhisper2.0_large"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    encoder = WhisperModel.from_pretrained(model_id, dtype=torch.float32).encoder.to("cuda").eval()
    vectors = []
    for path in tqdm(paths, desc="whisper"):
        audio, sample_rate = sf.read(path, dtype="float32")
        total, frames = 0, 0
        for start in range(0, len(audio), 30 * sample_rate):   # 30 s windows
            piece = audio[start:start + 30 * sample_rate]
            if len(piece) < sample_rate * 0.5:
                continue
            features = feature_extractor(piece, sampling_rate=sample_rate, return_tensors="pt").input_features.to("cuda")
            real_frames = int(np.ceil(len(piece) / 320))      # frames with real audio (50 per second)
            with torch.no_grad():
                layer = encoder(features, output_hidden_states=True).hidden_states[32]
            total = total + layer[0, :real_frames].sum(dim=0)
            frames += real_frames
        vectors.append((total / frames).cpu().numpy())
    return np.array(vectors)


def embed_phi4(paths):
    """Phi-4 audio encoder, layer 23: average over all frames of the (at most 40 s) pieces of each clip."""
    model_id = "microsoft/Phi-4-multimodal-instruct"
    prefix = "model.embed_tokens_extend.audio_embed.encoder."
    renames = [(r"feed_forward_(in|out)\.net\.0\.linear", r"feed_forward_\1.gate_up_proj"),
               (r"feed_forward_(in|out)\.net\.2", r"feed_forward_\1.down_proj"),
               (r"self_attn\.linear_(q|k|v)\.", r"self_attn.\1_proj."),
               (r"self_attn\.linear_out\.", r"self_attn.o_proj.")]
    # The repo's config is in the old (remote-code) format; converted by hand as in the official conversion script:
    # every value equals the transformers default except t5_bias_max_distance = 500.
    config = Phi4MultimodalAudioConfig(bias_max_distance=500)
    encoder = Phi4MultimodalAudioModel._from_config(config, attn_implementation="sdpa", dtype=torch.float32)
    weights = {}
    with safe_open(hf_hub_download(model_id, "model-00001-of-00003.safetensors"), framework="pt") as weight_file:
        for key in weight_file.keys():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                for pattern, replacement in renames:
                    new_key = re.sub(pattern, replacement, new_key)
                weights[new_key] = weight_file.get_tensor(key).float()
    encoder.load_state_dict(weights, strict=True)   # fails if any weight is missing or left over
    encoder = encoder.to("cuda").eval()
    # The encoder does not return the output of each layer, so a "hook" on every layer saves it while the clip
    # passes through: after one forward pass, layer_outputs[k] is the output of layer k + 1
    layer_outputs = []
    for layer in encoder.encoders:
        layer.register_forward_hook(lambda module, inputs, output: layer_outputs.append(output[0]))
    feature_extractor = Phi4MultimodalFeatureExtractor()
    vectors = []
    with torch.no_grad():
        for path in tqdm(paths, desc="phi4 audio"):
            audio, sample_rate = sf.read(path, dtype="float32")
            pieces = np.array_split(audio, int(np.ceil(len(audio) / (40 * sample_rate))))   # equal pieces <= 40 s
            total, frames = 0, 0
            for piece in pieces:
                features = feature_extractor(piece, sampling_rate=sample_rate, return_tensors="pt")["audio_input_features"]
                layer_outputs.clear()
                encoder(features.to("cuda"), None)
                layer = layer_outputs[22]            # layer 23 (counting from 1)
                total = total + layer.sum(dim=0)
                frames += layer.shape[0]
            vectors.append((total / frames).cpu().numpy())
    return np.array(vectors)


def embed_parakeet(paths, model_id, model_class, layer_number, processor_has_extractor):
    """Parakeet encoder (CTC or TDT), layer layer_number: average over all frames of the whole clip."""
    if processor_has_extractor:
        feature_extractor = AutoProcessor.from_pretrained(model_id).feature_extractor   # TDT keeps it in the processor
    else:
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
    encoder = model_class.from_pretrained(model_id, dtype=torch.float32).get_encoder().to("cuda").eval()
    vectors = []
    with torch.no_grad():
        for path in tqdm(paths, desc=model_id):
            audio, sample_rate = sf.read(path, dtype="float32")
            inputs = feature_extractor(audio, sampling_rate=sample_rate, return_tensors="pt").to("cuda")
            layer = encoder(**inputs, output_hidden_states=True).hidden_states[layer_number]
            vectors.append(layer[0].mean(dim=0).cpu().numpy())
    return np.array(vectors)


def embed_deberta(texts):
    """DeBERTa-v3-large, layer 21: average over all tokens of each transcript."""
    model_id = "microsoft/deberta-v3-large"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, dtype=torch.float32).to("cuda").eval()
    vectors = []
    with torch.no_grad():
        for text in tqdm(texts, desc="deberta-v3-large"):
            tokens = tokenizer(text, return_tensors="pt").to("cuda")
            layer = model(**tokens, output_hidden_states=True).hidden_states[21]
            vectors.append(layer[0].mean(dim=0).cpu().numpy())
    return np.array(vectors)


# ---- Clips (training clips scored 1-5 only) ----
train = pd.read_csv("output_submission/transcripts_train.csv")
train = train[train["label"] > 0].reset_index(drop=True)
test = pd.read_csv("data/test.csv")
test_transcripts = pd.read_csv("output_submission/transcripts_test.csv")
assert list(test_transcripts["filename"]) == list(test["filename"])   # same order, so text and audio rows match
paths = [os.path.join("data/train", f) for f in train["filename"]] + [os.path.join("data/test", f) for f in test["filename"]]
texts = [clean_text(t) for t in pd.concat([train["transcript"], test_transcripts["transcript"]]).fillna("")]
n_train = len(train)
print(f"Training clips: {n_train}  Test clips: {len(test)}")

# Each part: file name -> function that computes its vectors (run only when the file is not saved yet)
PARTS = {"whisper_l32": lambda: embed_whisper(paths),
         "phi4_audio_l23": lambda: embed_phi4(paths),
         "parakeet_ctc_l36": lambda: embed_parakeet(paths, "nvidia/parakeet-ctc-1.1b", ParakeetForCTC, 36, False),
         "parakeet_tdt_l20": lambda: embed_parakeet(paths, "nvidia/parakeet-tdt-0.6b-v3", ParakeetForTDT, 20, True),
         "deberta_v3_large_l21": lambda: embed_deberta(texts)}
os.makedirs(EMBEDDING_FOLDER, exist_ok=True)
for name, embed in PARTS.items():
    train_file = f"{EMBEDDING_FOLDER}/{name}_train.npy"
    test_file = f"{EMBEDDING_FOLDER}/{name}_test.npy"
    vectors = embed().astype(np.float32)   # training clips first, then test clips
    torch.cuda.empty_cache()
    print(f"{name}: {vectors.shape[0]} clips, {vectors.shape[1]} numbers each", flush=True)
    np.save(train_file, vectors[:n_train])
    np.save(test_file, vectors[n_train:])
