# GED-ELECTRA "error embedding" of every transcript: which kind of grammar errors the speaker makes.
#
# GED-ELECTRA (gotutiyan/token-ged-electra-large-bin) marks each word of a text as CORRECT or INCORRECT: its classifier
# head turns each word's last-layer embedding into P(INCORRECT).
# Error embedding = the words' last-layer embeddings averaged with weights P(INCORRECT): words that look wrong count
# most, so it describes the errors rather than the topic. (A plain average dilutes the few wrong words among ~100.)
# Speaker-grouped CV (frozen, Ridge): DeBERTa-large alone 0.596; joined with the error embedding 0.584.
# A word = a space-separated piece of the transcript; its P(INCORRECT) is taken from its first sub-token.
# Text: sound tags like [breath] removed; fillers [UH] and [UM] kept (as in the other scripts).
#
# Saves:  output_submission/embeddings/ged_error_embedding_train.npy   (training clips scored 1-5, order of transcripts_train.csv)
#         output_submission/embeddings/ged_error_embedding_test.npy    (test clips, order of transcripts_test.csv)
# Run:    .venv/Scripts/python src/error_embedding.py

import os
import re

import truststore
truststore.inject_into_ssl()  # use the Windows certificate store, otherwise the model download fails

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoTokenizer

MODEL_ID = "gotutiyan/token-ged-electra-large-bin"   # licence: research/education only
EMBEDDING_FOLDER = "output_submission/embeddings"


def clean_text(text):
    """Remove sound tags like [breath] or [laughter] (lowercase); keep the fillers [UH] and [UM]."""
    text = re.sub(r"\[[a-z]+\]", " ", text)
    return " ".join(text.split())


def error_embedding(text, tokenizer, model):
    """Average of the words' last-layer embeddings, weighted by each word's P(INCORRECT)."""
    words = text.split()
    tokens = tokenizer(words, is_split_into_words=True, return_tensors="pt").to("cuda")
    with torch.no_grad():
        output = model(**tokens, output_hidden_states=True)
    word_ids = tokens.word_ids()                                    # which word each sub-token belongs to
    first = [word_ids.index(w) for w in range(len(words))]          # first sub-token of each word
    p_incorrect = torch.softmax(output.logits[0], dim=-1)[first, 1]  # label 1 = INCORRECT
    word_embeddings = output.hidden_states[-1][0, first]            # shape: (words, 1024)
    weighted = (p_incorrect[:, None] * word_embeddings).sum(dim=0) / p_incorrect.sum()
    return weighted.cpu().numpy()


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForTokenClassification.from_pretrained(MODEL_ID, dtype=torch.float32).to("cuda").eval()
os.makedirs(EMBEDDING_FOLDER, exist_ok=True)

train = pd.read_csv("output_submission/transcripts_train.csv")
train = train[train["label"] > 0].reset_index(drop=True)
test = pd.read_csv("output_submission/transcripts_test.csv")
for name, table in [("train", train), ("test", test)]:
    texts = [clean_text(t) for t in table["transcript"].fillna("")]
    if min(len(t.split()) for t in texts) == 0:
        raise ValueError(f"An empty transcript in {name}")   # would need a decision on what to use instead
    embeddings = np.array([error_embedding(t, tokenizer, model) for t in tqdm(texts, desc=name)])
    np.save(os.path.join(EMBEDDING_FOLDER, f"ged_error_embedding_{name}.npy"), embeddings)
    print(name, embeddings.shape)
