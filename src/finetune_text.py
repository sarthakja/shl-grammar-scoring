# Fine-tune a DeBERTa text model on the transcripts to predict the grammar score (step 5 of the notebook).
#
# Model:  transcript -> DeBERTa -> average of the top layer's token vectors -> one output neuron (the score).
#         Dropout is turned off: it makes the outputs noisier during training, which hurts a regression output.
# Models: deberta_v3_large  (microsoft/deberta-v3-large, all 24 layers, 435M weights trained; the default)
#         deberta_v2_xlarge (microsoft/deberta-v2-xlarge, 900M weights): cut after layer 12 of 24 (its best frozen
#                           layer), and its word-embedding table (197M weights) is kept frozen, so 348M weights are
#                           trained. Training all 900M would need ~15 GB for weights and optimiser alone (the GPU
#                           has 16 GB); with the cut, peak GPU memory is 9.4 GB.
# Target: the score standardised with the training clips' mean and standard deviation; loss = mean squared error.
# Training: AdamW, learning rate 1e-5 for DeBERTa (adjust, not overwrite) and 1e-3 for the new output neuron (it
#         starts from random numbers and would barely move at 1e-5), weight decay 0.01. Batches of 8 clips, shuffled
#         every epoch; 6 epochs; the learning rate rises from 0 over the first 10% of steps, then falls linearly to 0.
# CV:     the same 5 speaker-grouped folds as the notebook (output_submission/speaker_groups.csv, from src/speakers.py).
#         After EVERY epoch, each fold model predicts its validation, training and test clips.
#         One epoch is chosen for all folds (lowest mean CV RMSE), not the best epoch per fold (that would make the
#         CV numbers look better than they are). Test prediction = average of the 5 fold models at that epoch.
# Only clips scored 1-5 are used (score-0 clips are handled by the noise rule). About 10 min per model on an RTX 5080 laptop GPU.
#
# Saves (name = finetune_deberta_v3_large or finetune_deberta_v2_xlarge):
#         output_submission/<name>_results.csv           (per epoch: CV RMSE and Pearson, mean and std over folds; training RMSE)
#         output_submission/<name>_epochs.csv            (training loss and validation RMSE per fold and epoch, written as it goes)
#         output_submission/<name>_cv_predictions.csv    (each training clip's cross-validated prediction at the chosen epoch,
#                                               by the fold model that did not train on it; column "prediction")
#         output_submission/<name>_test_predictions.csv  (each test clip: average prediction of the 5 fold models at the chosen
#                                               epoch; column "average")
#         The epoch is chosen only after all 5 folds are trained, and the models are not saved (too large), so while
#         training, the predictions after every epoch are kept in memory; only the chosen epoch's are saved.
#         output_submission/<name>_train_predictions.csv (each training clip: average prediction of the 4 fold models trained on
#                                               it, at the chosen epoch; for the training RMSE)
# Run:    .venv/Scripts/python src/finetune_text.py                            (DeBERTa-v3-large)
#         .venv/Scripts/python src/finetune_text.py --model=deberta_v2_xlarge  (DeBERTa-v2-xlarge)

import os
import re
import sys

import truststore
truststore.inject_into_ssl()  # use the Windows certificate store, otherwise the model download fails

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr
from sklearn.model_selection import GroupKFold
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

# ---- Settings ----
# name: (Hugging Face model id, layers kept, word embeddings frozen)
MODELS = {"deberta_v3_large": ("microsoft/deberta-v3-large", 24, False),
          "deberta_v2_xlarge": ("microsoft/deberta-v2-xlarge", 12, True)}
MODEL = "deberta_v3_large"
for argument in sys.argv:
    if argument.startswith("--model="):
        MODEL = argument.split("=")[1]
MODEL_ID, KEEP_LAYERS, FREEZE_WORD_EMBEDDINGS = MODELS[MODEL]
NAME = "finetune_" + MODEL
SPEAKERS_CSV = "output_submission/speaker_groups.csv"

SEED = 42                  # random start of the output neuron and the clip order
EPOCHS = 6
BATCH_SIZE = 8
ENCODER_LEARNING_RATE = 1e-5
OUTPUT_LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.01        # AdamW: every step, each trained weight shrinks by learning rate x WEIGHT_DECAY
WARMUP_FRACTION = 0.1      # learning rates rise from 0 over the first 10% of steps, then fall to 0
MAX_TOKENS = 512           # longest input DeBERTa accepts (our longest transcript: ~250 tokens)
N_FOLDS = 5


def clean_text(text):
    """Remove sound tags like [breath] or [laughter] (lowercase); keep the fillers [UH] and [UM]."""
    text = re.sub(r"\[[a-z]+\]", " ", text)
    return " ".join(text.split())


def rmse(true, predicted):
    """Root mean squared error."""
    return np.sqrt(np.mean((true - predicted) ** 2))


class TextScoreModel(torch.nn.Module):
    """DeBERTa; the average of its top layer's token vectors -> one output neuron."""

    def __init__(self):
        super().__init__()
        # float32: the checkpoint is saved in float16, which would turn the weights into NaN after one training step
        self.encoder = AutoModel.from_pretrained(MODEL_ID, dtype=torch.float32,
                                                 hidden_dropout_prob=0.0, attention_probs_dropout_prob=0.0)
        self.encoder.encoder.layer = self.encoder.encoder.layer[:KEEP_LAYERS]   # top layer = layer KEEP_LAYERS
        if FREEZE_WORD_EMBEDDINGS:
            self.encoder.get_input_embeddings().weight.requires_grad = False
        self.output_layer = torch.nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        token_vectors = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).float()                  # 1 for real tokens, 0 for padding
        average = (token_vectors.float() * mask).sum(dim=1) / mask.sum(dim=1)
        return self.output_layer(average)[:, 0]


def make_batch(index, table_texts):
    """Tokenised transcripts, padded to the longest in the batch."""
    return tokenizer([table_texts[i] for i in index], padding=True, return_tensors="pt").to("cuda")


def predict(model, table_texts, score_mean, score_std):
    """Predicted scores (1-5)."""
    model.eval()
    predictions = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(table_texts), BATCH_SIZE):
            index = np.arange(start, min(start + BATCH_SIZE, len(table_texts)))
            tokens = make_batch(index, table_texts)
            predictions.extend(model(tokens["input_ids"], tokens["attention_mask"]).float().cpu().numpy())
    return np.clip(np.array(predictions) * score_std + score_mean, 1, 5)   # undo the standardisation


def train(fold, train_index, validation_index):
    """Fine-tune a fresh model. After every epoch: predict validation, training and test clips; log to the epochs CSV."""
    torch.manual_seed(SEED)   # same starting output neuron for every fold
    model = TextScoreModel().to("cuda")
    if fold == 1:
        trained = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  trained weights: {trained / 1e6:.1f} million")
    score_mean = y[train_index].mean()
    score_std = y[train_index].std()
    targets = (y[train_index] - score_mean) / score_std
    train_texts = texts[train_index]

    encoder_weights = [p for p in model.encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": encoder_weights, "lr": ENCODER_LEARNING_RATE},
        {"params": model.output_layer.parameters(), "lr": OUTPUT_LEARNING_RATE}], weight_decay=WEIGHT_DECAY)
    steps_per_epoch = int(np.ceil(len(train_index) / BATCH_SIZE))
    total_steps = EPOCHS * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, int(WARMUP_FRACTION * total_steps), total_steps)

    random_generator = np.random.default_rng(SEED)
    validation_by_epoch, train_by_epoch, test_by_epoch = [], [], []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        order = random_generator.permutation(len(train_index))   # new random clip order every epoch
        losses = []
        for start in range(0, len(order), BATCH_SIZE):
            index = order[start:start + BATCH_SIZE]
            tokens = make_batch(index, train_texts)
            target = torch.tensor(targets[index], dtype=torch.float32, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):   # faster, less memory; weights stay float32
                output = model(tokens["input_ids"], tokens["attention_mask"])
            loss = torch.nn.functional.mse_loss(output.float(), target)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        validation_by_epoch.append(predict(model, texts[validation_index], score_mean, score_std))
        train_by_epoch.append(predict(model, train_texts, score_mean, score_std))
        test_by_epoch.append(predict(model, test_texts, score_mean, score_std))
        row = {"fold": fold, "epoch": epoch, "train_loss": np.mean(losses),
               "validation_rmse": rmse(y[validation_index], validation_by_epoch[-1])}
        print(f"  epoch {epoch}: train loss {row['train_loss']:.3f}, validation RMSE {row['validation_rmse']:.3f}")
        pd.DataFrame([row]).round(4).to_csv(EPOCHS_CSV, mode="a", header=not os.path.exists(EPOCHS_CSV), index=False)

    del model
    torch.cuda.empty_cache()
    return validation_by_epoch, train_by_epoch, test_by_epoch


# ---- Load transcripts (training clips scored 1-5 only) ----
clips = pd.read_csv("output_submission/transcripts_train.csv")
clips = clips[clips["label"] > 0].reset_index(drop=True)
texts = np.array([clean_text(t) for t in clips["transcript"].fillna("")], dtype=object)
y = clips["label"].to_numpy()
test = pd.read_csv("output_submission/transcripts_test.csv")
test_texts = np.array([clean_text(t) for t in test["transcript"].fillna("")], dtype=object)
print("Model:", MODEL_ID, " Training clips:", len(clips), " Test clips:", len(test))

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
longest = max(len(tokenizer(t)["input_ids"]) for t in list(texts) + list(test_texts))
print("Longest transcript:", longest, "tokens")
if longest > MAX_TOKENS:
    raise ValueError("A transcript is longer than the model accepts")   # never cut a transcript

# Same folds as the notebook; each speaker's clips stay in one fold
speakers = pd.read_csv(SPEAKERS_CSV).set_index("filename").loc[clips["filename"], "speaker"].to_numpy()
folds = list(GroupKFold(n_splits=N_FOLDS).split(clips, groups=speakers))
EPOCHS_CSV = f"output_submission/{NAME}_epochs.csv"
if os.path.exists(EPOCHS_CSV):
    os.remove(EPOCHS_CSV)   # start a fresh epoch log for this run


# ---- 5-fold cross-validation ----
cv_predictions = np.zeros((EPOCHS, len(y)))      # row e: each clip's cross-validated prediction after epoch e+1
fold_rmses = np.zeros((EPOCHS, N_FOLDS))
fold_pearsons = np.zeros((EPOCHS, N_FOLDS))
train_rmses = np.zeros((EPOCHS, N_FOLDS))
train_prediction_sums = np.zeros((EPOCHS, len(y)))   # each clip: sum of the predictions of the 4 folds that trained on it
test_prediction_sums = np.zeros((EPOCHS, len(test)))  # each test clip: sum of the predictions of the 5 fold models
for fold, (train_index, validation_index) in enumerate(folds, start=1):
    print(f"\nFold {fold} of {N_FOLDS}")
    validation_by_epoch, train_by_epoch, test_by_epoch = train(fold, train_index, validation_index)
    for e in range(EPOCHS):
        cv_predictions[e, validation_index] = validation_by_epoch[e]
        fold_rmses[e, fold - 1] = rmse(y[validation_index], validation_by_epoch[e])
        fold_pearsons[e, fold - 1] = pearsonr(y[validation_index], validation_by_epoch[e])[0]
        train_rmses[e, fold - 1] = rmse(y[train_index], train_by_epoch[e])
        train_prediction_sums[e, train_index] += train_by_epoch[e]
        test_prediction_sums[e] += test_by_epoch[e]


# ---- Results ----
results = pd.DataFrame({"epoch": np.arange(1, EPOCHS + 1),
                        "cv_rmse_mean": fold_rmses.mean(axis=1), "cv_rmse_std": fold_rmses.std(axis=1),
                        "cv_pearson_mean": fold_pearsons.mean(axis=1), "cv_pearson_std": fold_pearsons.std(axis=1),
                        "train_rmse": train_rmses.mean(axis=1)}).round(3)
print("\n", results.to_string(index=False))
results.to_csv(f"output_submission/{NAME}_results.csv", index=False)

best_epoch = int(results.loc[results["cv_rmse_mean"].idxmin(), "epoch"])   # one epoch for all folds
print(f"\nChosen epoch (lowest mean CV RMSE): {best_epoch}")

# Save the predictions at the chosen epoch only
cv_table = clips[["filename", "label"]].copy()
cv_table["prediction"] = cv_predictions[best_epoch - 1]
cv_table.to_csv(f"output_submission/{NAME}_cv_predictions.csv", index=False)

train_table = clips[["filename", "label"]].copy()
train_table["average"] = train_prediction_sums[best_epoch - 1] / (N_FOLDS - 1)   # for the training RMSE
train_table.to_csv(f"output_submission/{NAME}_train_predictions.csv", index=False)

test_table = test[["filename"]].copy()
test_table["average"] = test_prediction_sums[best_epoch - 1] / N_FOLDS
test_table.to_csv(f"output_submission/{NAME}_test_predictions.csv", index=False)
print("Saved output_submission/" + NAME + "_*.csv")
