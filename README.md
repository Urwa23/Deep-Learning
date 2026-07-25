# AI-Generated Text Detection & Interpretability

Fine-tune DistilBERT to distinguish human-written vs. LLM-generated text, use
Captum to see what it's actually keying on, and test whether a classifier
trained on ChatGPT text generalizes to Claude text.

## Research question

Train a binary classifier (human vs. AI) on HC3, apply saliency / Integrated
Gradients to explain its predictions, and test cross-model generalization by
evaluating the ChatGPT-trained model on Claude-generated text.

## Setup

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

All local work uses this `.venv` — do not install into the global Python.
`torch` here is CPU-only; GPU training happens on Colab (see below).

## Pipeline

| Step | Where | Runs |
|---|---|---|
| Data exploration (domain sizes, sample read-through, length distributions) | `notebooks/01_data_exploration.ipynb` | local |
| Build binary train/val/test splits (length-controlled + raw variants) | `notebooks/02_data_prep.ipynb`, `scripts/prepare_data.py` | local |
| Fine-tune DistilBERT (both variants) | `notebooks/03_finetune_distilbert.ipynb` | **Colab (GPU)** |
| Interpretability (Captum: Integrated Gradients + Saliency) | `notebooks/04_interpretability.ipynb` | local, after downloading checkpoints |
| Cross-model generalization test (ChatGPT-trained model vs. Claude text) | `notebooks/05_cross_model_test.ipynb` | local, after downloading checkpoints |

### Why training runs on Colab

Local CPU benchmarking showed DistilBERT trains at ~1 sample/sec at
seq_len=256 on this machine — full-dataset training (23k examples x 3 epochs
x 2 length variants) would take 18+ hours locally. Colab's free T4 GPU cuts
this to well under an hour per variant. `03_finetune_distilbert.ipynb` is
self-contained (re-downloads and reprocesses HC3 directly) so it can be
opened and run on Colab without uploading any local files.

After training, download the `ai_text_detection_checkpoints/` folder from
Google Drive into `results/checkpoints/` locally (matching the
`length_controlled/final/` and `raw/final/` structure) before running
notebooks 04 and 05.

## Key methodology decisions

- **Domain scope**: HC3's `reddit_eli5` subset only (17k rows, the largest
  and most general-topic domain) — kept the 2-week scope tight.
- **Length control**: exploration showed ChatGPT answers are systematically
  longer and much lower-variance than human answers (median ~173 vs ~82
  words, std ~62 vs ~164 on reddit_eli5) — a classifier could cheat on length
  alone. The main model trains on human/chatgpt pairs truncated to a matched
  word budget (`length_controlled`); a second model trains on original
  lengths (`raw`) purely to demonstrate the shortcut via interpretability.
  Truncation is **sentence-boundary-aware** (`_truncate_to_words_sentence_aware`
  in `src/data_utils.py`): an earlier hard word-count cutoff left 63.7% of
  human answers ending cleanly vs. only 24.3% of (truncated) chatgpt answers —
  "ends mid-sentence" would have been almost as strong a shortcut as the
  length signal it was meant to remove. Truncating at sentence boundaries
  brought that to 91.8%/99.5%, at the cost of the two sides of a pair no
  longer having exactly equal word counts (residual gap ~11-14 words median,
  vs. the original ~90-word gap).
- **Escaped-newline artifact**: interpretability analysis (`04_interpretability.ipynb`)
  on the first trained checkpoints found a literal backslash `\` was the single
  strongest token pushing predictions toward "AI" — traced back to HC3's raw
  source data: 10.6% of chatgpt_answers contain a literal two-character `\n`
  marker (paragraph breaks stored as text, never converted to whitespace) vs.
  0.0% of human_answers. That's a data-collection artifact, not style, and a
  cleaner shortcut than the length one. `_clean_text()` in `src/data_utils.py`
  strips it before any downstream processing; both checkpoints were retrained
  after the fix. The pre-fix checkpoints and their interpretability results are
  kept in `results/checkpoints/_pre_artifact_fix/` for reference.
- **Train/val/test split** is grouped by question (`pair_id` in
  `src/data_utils.py`) so a question's human and chatgpt answers always land
  in the same split.
- **Cross-model test data**: uses the pre-existing Claude-essay subset
  (`darragh_claude_v6`/`v7`) of the Kaggle DAIGT V2 dataset rather than
  generating new Claude text, for speed. This swaps *both* the generator
  model (ChatGPT -> Claude) and the domain (Reddit Q&A -> persuasive student
  essays) at once — treated as a limitation, not a clean model-only
  comparison, in the write-up.

## Project layout

```
data/{raw,interim,processed}/     — processed/{length_controlled,raw}/{train,val,test}.parquet
notebooks/                        — 01 exploration, 02 data prep, 03 Colab training,
                                     04 interpretability, 05 cross-model test
src/
  data_utils.py                   — load_hc3(), flatten_to_binary(), group_train_val_test_split()
  model.py                        — DistilBERT tokenizer/model loaders
  train.py                        — local CPU training script (smoke-testing only; real
                                     training runs via notebook 03 on Colab)
  interpret.py                    — Captum Integrated Gradients / Saliency helpers
scripts/
  explore_hc3.py                  — script form of notebook 01
  prepare_data.py                 — script form of notebook 02
results/{figures,checkpoints}/, results/metrics.json
report/                           — proposal.md, final_writeup.md
```
