# AI-Generated Text Detection & Interpretability

## 1. Research question

Can a fine-tuned classifier distinguish human-written text from LLM-generated text on genuine stylistic grounds, what specifically does it key on, and does a detector trained on one LLM's output generalize to a different LLM's output? We fine-tune DistilBERT as a binary classifier on human vs. ChatGPT text, apply gradient-based interpretability (Captum) to identify what the model actually relies on, and test whether the resulting detector transfers to Claude-generated text.

This builds on Guo et al. (2023), *"How Close is ChatGPT to Human Experts? Comparison Corpus, Evaluation, and Detection,"* which introduced the HC3 dataset used here, ran a human Turing-test evaluation, and fine-tuned RoBERTa detectors reaching near-ceiling accuracy. That paper characterizes human-vs-ChatGPT differences through corpus-level statistics (formality, lexical diversity, topic drift) and does not examine a trained classifier's internal decision process, does not test whether detection accuracy survives removing surface-level shortcuts (length, formatting artifacts), and does not test transfer to a different generator model. Those three questions are the focus here.

## 2. Dataset

**HC3** (Human ChatGPT Comparison Corpus, `Hello-SimpleAI/HC3`): 24,322 rows across five domains, each a `question` paired with one or more `human_answers` and one or more `chatgpt_answers`, topic-matched by construction.

| Domain | Rows |
|---|---|
| reddit_eli5 | 17,112 |
| finance | 3,933 |
| medicine | 1,248 |
| open_qa | 1,187 |
| wiki_csai | 842 |

We scoped to **reddit_eli5 only** — the largest and most general-topic domain — to keep the project tractable within a two-week timeline rather than modeling five domains at once.

## 3. Data preparation

Each question's first human answer and first ChatGPT answer are paired 1:1 (avoiding class imbalance from questions with multiple human answers) and flattened into a binary-labeled dataset (0 = human, 1 = AI). After filtering empty/near-empty answers, this yields **33,314 examples (16,657 / 16,657, exactly balanced)**, split 70/15/15 into train/val/test (23,318 / 4,998 / 4,998), **grouped by question** so a topic never appears on both sides of a split.

### 3.1 The length shortcut

Exploratory analysis showed ChatGPT answers are systematically longer and far lower-variance than human answers:

| | median words | mean words | std |
|---|---|---|---|
| Human | 82 | 133.3 | 166.8 |
| ChatGPT | 173 | 174.9 | 53.4 |

A classifier could reach high accuracy by learning "length ≈ 150–220 words, low variance → AI" without any stylistic reasoning. We address this by training two variants:

- **`length_controlled`**: each human/chatgpt pair truncated to a shared word budget, `min(human_words, chatgpt_words, 200)`.
- **`raw`**: original, uncontrolled lengths — kept specifically as an ablation to demonstrate the shortcut's effect.

**A truncation bug, caught and fixed.** The first implementation truncated at a hard word-count cutoff, which cut ChatGPT's answer (usually the longer side) off mid-sentence in most pairs. This left "does the text end abruptly" as a shortcut almost as strong as the length signal it replaced — 63.7% of human answers ended cleanly vs. only 24.3% of truncated ChatGPT answers. We replaced this with **sentence-boundary-aware truncation**: walk sentences front-to-back, greedily keeping whole sentences until the next one would exceed the word budget (never cutting mid-sentence, at the cost of the two sides of a pair no longer having exactly equal word counts — residual gap ~11–14 words median, vs. the original ~90-word gap). This raised the clean-ending rate to 91.8% / 99.5%.

### 3.2 The escaped-newline artifact

After the first full training run, Captum interpretability analysis (Section 5) found a lone backslash `\` was the single strongest token driving predictions toward "AI" — stronger than any actual word. Tracing this to HC3's raw source data confirmed it was a data-collection artifact, not style: **10.6% of chatgpt_answers contain a literal two-character `\n` marker** (a paragraph break stored as text and never converted to whitespace) **vs. 0.0% of human_answers**. This is a cleaner, more exploitable shortcut than the length signal, and specific to how HC3 was collected — a live ChatGPT completion would never contain this literal marker. We added a cleaning step stripping this marker before any downstream processing, regenerated the data, and retrained both variants. The artifact-affected checkpoints and their interpretability results were kept (`results/checkpoints/_pre_artifact_fix/`) as a documented before/after comparison rather than discarded.

## 4. Model and training

**DistilBERT** (`distilbert-base-uncased`), fine-tuned as a binary sequence classifier — full fine-tuning, not inference on a frozen pretrained model.

**Compute note.** Local hardware is CPU-only; benchmarking showed ~1 sample/sec at seq_len=256, making the full training plan (23k examples × 3 epochs × 2 variants) an 18+ hour job locally. Training was moved to Google Colab (T4 GPU), using a self-contained notebook that re-downloads and reprocesses HC3 directly.

**Final results** (post both fixes):

| Variant | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| `length_controlled` | 98.64% | 97.54% | 99.80% | 98.66% |
| `raw` | 99.68% | 99.44% | 99.92% | 99.68% |

Both variants reach near-ceiling accuracy. `raw` outperforms `length_controlled` by roughly one point — consistent with `raw` still having access to the length shortcut, which `length_controlled` was specifically designed to remove. (Before the escaped-newline fix, both variants scored slightly higher — 98.99% / 99.82% — the small drop after removing both shortcuts is expected and reassuring: it indicates genuine signal remains dominant rather than shortcut-inflated accuracy.)

## 5. Interpretability

We applied Captum **Integrated Gradients** and **Saliency** directly to the fine-tuned DistilBERT models (not corpus-level statistics), attributing each prediction back to input tokens.

**Aggregate vocabulary signal** (averaged attribution across 100 test examples per model, after both fixes):

- `length_controlled` — pushes toward **AI**: `reasons`, `process`, `especially`, `why`, `caused`, `known`, `designed`, `depending`, `legal`, `called`. Pushes toward **HUMAN**: `generally`, `years`, `had`, `was`, `its`, `him`, `her`.
- `raw` — pushes toward **AI**: `simply`, `always`, `additionally`, `potentially`, `overall`, `because`, `imagine`. Pushes toward **HUMAN**: `most`, `years`, `;`, `here`, `very`, `cash`, `him`.

This is a coherent, genuine stylistic signal: the "AI" side consistently surfaces formal, causal/explanatory connectives and hedging language, matching Guo et al.'s qualitative characterization of ChatGPT answers as formal and definition-first. The "HUMAN" side surfaces personal pronouns and past-tense narrative markers, consistent with informal, first-person Reddit writing. One caveat: `youtube` appears among AI-pushing tokens, likely reflecting topic-specific content (some questions are literally about YouTube) rather than style — a content/style ambiguity worth flagging rather than a defect.

**Quantifying the length shortcut.** Correlation between word count and predicted P(AI): **-0.034** for `length_controlled` vs. **+0.088** for `raw`. Small in absolute terms — likely dampened because the tokenizer's 256-token cap limits how much of a very long `raw` answer the model actually observes — but directionally exactly as designed: length control removes the correlation almost entirely, while `raw` retains a small positive one.

## 6. Cross-model generalization: does a ChatGPT detector catch Claude?

The `length_controlled` model — trained exclusively on human vs. ChatGPT text from reddit_eli5 — was evaluated on the Kaggle **DAIGT V2** dataset's Claude-essay subset (`darragh_claude_v6`/`v7`, ~2,000 Claude-written persuasive essays) against its human student-essay subset (`persuade_corpus`), on a balanced 4,000-example set.

| Setting | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| In-domain (HC3, ChatGPT, reddit_eli5) | 98.64% | 97.54% | 99.80% | 98.66% |
| Cross-model + cross-domain (DAIGT, Claude, essays) | 75.53% | 80.37% | 67.55% | 73.40% |

Accuracy drops substantially — but stays well above the 50% chance baseline, indicating some transferable signal survives. More interesting is the asymmetry in the confusion matrix:

- **Human-essay recall (specificity): 83.5%** — still fairly reliable at recognizing human writing, even in an unfamiliar domain.
- **Claude-essay recall (sensitivity): 67.5%** — notably worse at recognizing Claude's writing as AI; about a third of Claude essays are misclassified as human.

This asymmetry is consistent with the interpretability findings: the model's learned "AI" markers are specific formal ChatGPT-style connectives. If Claude hedges and explains differently than ChatGPT, that is exactly the failure mode this would produce — the classifier has partly learned "ChatGPT's style" rather than a generator-agnostic "AI-generated" signal.

**Important caveat.** This test changes both the generator model (ChatGPT → Claude) *and* the domain (Reddit Q&A → persuasive essays) simultaneously. The accuracy drop cannot be cleanly attributed to one factor alone — it is suggestive of model-specific overfitting, not conclusive proof of it. Isolating model-shift alone would require Claude-generated text on matched reddit_eli5-style questions, which we did not generate due to time/cost constraints (see Limitations).

## 7. Discussion

Three findings stand out:

1. **Two independent shortcuts were found and fixed before trusting the results.** A naive length-matching truncation introduced a new "ends mid-sentence" tell nearly as strong as the length signal it replaced; a source-data formatting quirk (escaped newlines) was, at one point, the single strongest signal either model relied on. Both were identified through direct inspection of what the truncation produced and what the interpretability tooling attributed weight to — not assumed away. This matters for the central claim of the project: that the reported accuracy reflects real stylistic detection rather than an artifact of preprocessing or data collection.
2. **After both fixes, genuine and coherent stylistic signal remains**, and it looks like exactly the kind of formal/explanatory-vs-informal/personal register difference the original HC3 paper described qualitatively — now demonstrated mechanistically, at the level of what a trained classifier's gradients actually attend to.
3. **That signal does not fully generalize to a different LLM.** The sensitivity/specificity asymmetry under cross-model testing suggests the classifier learned something closer to "ChatGPT's style" than "AI-generated style" in general — a meaningfully different and more cautious claim than "this detector catches AI text."

## 8. Limitations

- **Single domain** (reddit_eli5 only); results may not hold across HC3's other four domains (finance, medicine, open_qa, wiki_csai).
- **Cross-model test confounds domain and model shift simultaneously** — the accuracy drop cannot be attributed to Claude-vs-ChatGPT alone.
- **DistilBERT, not the larger RoBERTa** used in the original HC3 paper — a compute-driven scope reduction (CPU-only local hardware, GPU training moved to Colab's free tier).
- **Residual length variance** within `length_controlled` pairs (~11–14 words median) after sentence-aware truncation — smaller than the original confound, but not perfectly zero.
- **Interpretability findings are correlational**, not a formal causal account of model behavior; token attribution highlights what correlates with a prediction, not a guaranteed mechanism.
- **One topic-content leakage instance observed** (`youtube` as an AI-pushing token) suggests some residual conflation of content and style that a larger/more diverse sample might further disentangle.

## 9. Conclusion

A DistilBERT classifier fine-tuned on human-vs-ChatGPT text reaches ~98–99% in-domain accuracy, and interpretability analysis shows this is substantially driven by genuine stylistic register differences (formal/explanatory vs. informal/personal) rather than superficial artifacts — but only after two independent shortcuts (a truncation artifact and a data-collection artifact) were identified and removed. When evaluated cross-model on Claude-generated text in a different domain, accuracy drops to ~76%, with a pronounced asymmetry suggesting the model's notion of "AI-generated" is partly ChatGPT-specific rather than fully generator-agnostic. The practical implication: AI-text detectors validated only against their training generator should not be assumed to generalize to other LLMs without direct testing.
