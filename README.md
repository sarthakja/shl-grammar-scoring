# Grammar Scoring Engine for Spoken English

Predicts a 1-5 grammar score for spoken English audio clips. The report (approach, preprocessing, pipeline architecture, evaluation results, including the training RMSE) and the code are in **`grammar_scoring.ipynb`**.

Results: training RMSE 0.289, speaker-grouped 5-fold CV RMSE 0.516, public leaderboard RMSE 0.3255.

## Files

| Path | What it is |
|---|---|
| `grammar_scoring.ipynb` | Report and pipeline: runs every step, from the data to the submission |
| `src/` | The code files (some require a 16 GB GPU) used by the notebook: `snr.py` (noise level), `transcribe.py` (transcripts), `speakers.py` (speaker groups for CV), `error_embedding.py` (GED error embedding), `frozen_embeddings.py` (audio and text embeddings), `finetune_text.py` (fine-tuned DeBERTa models) |
| `output_submission/` | Everything the pipeline produces, from the last full run: transcripts, speaker groups, embeddings, fine-tuning results, figures, and the submission `submission_blend_ridge.csv` |
| `requirements.txt` | Python packages, with the versions used |

## Setup

1. **Python 3.14** and an **NVIDIA GPU with 16 GB** (only needed to re-run the expensive steps; the rest runs on CPU).
2. Install the packages (PyTorch with CUDA 12.8 comes from the PyTorch package index, already listed in the file):
   ```
   python -m venv .venv
   .venv/Scripts/activate        # Windows; on Linux/macOS: source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. **Data** (not in this repository, ~1.3 GB): download the competition data and place it as:
   ```
   data/train.csv      (filename, label)
   data/test.csv       (filename; any other column is ignored)
   data/train/*.wav
   data/test/*.wav
   ```

## Running

Open `grammar_scoring.ipynb`, with the environment above as the kernel, and run the cells from top to bottom.

* **Expensive steps** (transcription, embeddings, fine-tuning; about 2 h on an RTX 5080 laptop GPU) run only if `RUN_EXPENSIVE_STEPS = True` in the first code cell. With `False` (the default), they are skipped and the notebook uses their saved outputs in `output_submission/`; the rest takes a few minutes.
* The pretrained models (CrisperWhisper 2.0, Phi-4-multimodal, Parakeet, DeBERTa, WavLM, GED-ELECTRA) are downloaded from Hugging Face on first use (tens of GB).
* **Reproducibility:** GPU computations are not bit-for-bit repeatable, so re-running the expensive steps gives very close but not identical numbers (a full re-run gave predictions correlated 0.99999 with the original submission).

## Figures in the report

The figures in the notebook's Approach section are image files, linked by a relative path (`output_submission/figures/*.png`); the notebook writes them when it runs. They show when the notebook is opened from the repository folder (Jupyter, VS Code). Some viewers, possibly including GitHub's notebook preview, do not display images linked by a relative path; the same figures are also drawn by the code cells further down the notebook, and the image files themselves are in `output_submission/figures/`.
