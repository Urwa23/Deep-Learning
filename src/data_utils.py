"""
Loading and preprocessing for HC3 (Human ChatGPT Comparison Corpus).

`datasets.load_dataset("Hello-SimpleAI/HC3")` no longer works because the
HF `datasets` library (v4+) dropped support for HC3's loading-script format.
We bypass it: download the raw all.jsonl via huggingface_hub and parse it
by hand.

Exploration (scripts/explore_hc3.py) showed ChatGPT answers are systematically
longer and lower-variance than human answers (median ~173 vs ~82 words on
reddit_eli5), which would let a classifier cheat on length instead of style.
flatten_to_binary() offers a `length_control` option that truncates each
human/chatgpt pair to a shared word count to remove that shortcut.
"""
import json
import re

import pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.model_selection import GroupShuffleSplit

HC3_REPO_ID = "Hello-SimpleAI/HC3"
HC3_FILENAME = "all.jsonl"


def load_hc3() -> pd.DataFrame:
    """Download (or reuse cached) HC3 all.jsonl and return it as a raw DataFrame.

    Columns: question, human_answers (list[str]), chatgpt_answers (list[str]),
    index, source.
    """
    path = hf_hub_download(repo_id=HC3_REPO_ID, filename=HC3_FILENAME, repo_type="dataset")
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


_ESCAPED_NEWLINE_RE = re.compile(r"\\r\\n|\\n|\\r")
_WHITESPACE_RE = re.compile(r"\s+")


def _clean_text(text: str) -> str:
    """Normalize literal escaped-newline markers left over from HC3's source data.

    10.6% of raw chatgpt_answers contain a literal two-character "\\n" (paragraph
    break stored as text rather than converted to whitespace), vs. 0.0% of
    human_answers. That's a data-collection artifact, not a style difference,
    but it's a clean enough signal that a classifier finds it before genuine
    style - interpretability analysis showed it was the single strongest
    attributed token pushing predictions toward "AI". Strip it before any
    downstream processing so it can't be learned as a shortcut.
    """
    text = _ESCAPED_NEWLINE_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _truncate_to_words(text: str, n_words: int) -> str:
    words = text.split()
    return " ".join(words[:n_words])


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _truncate_to_words_sentence_aware(text: str, n_words: int) -> str:
    """Truncate to the last complete sentence at or before the n_words budget.

    A hard word-count cutoff (_truncate_to_words) leaves a systematic tell:
    the truncated side almost always ends mid-sentence while the untouched
    (shorter) side usually ends cleanly, which a classifier can pick up on
    as easily as it could pick up on raw length. This always keeps whole
    sentences, at the cost of two things: the two sides of a pair no longer
    have exactly equal word counts (off by up to one sentence), and if the
    first sentence alone exceeds n_words it is still kept in full (a small
    overshoot beats reintroducing a mid-sentence cut).
    """
    sentences = [s for s in _SENTENCE_SPLIT_RE.split(text.strip()) if s]
    kept_words = []
    for sentence in sentences:
        sentence_words = sentence.split()
        if kept_words and len(kept_words) + len(sentence_words) > n_words:
            break
        kept_words.extend(sentence_words)
        if len(kept_words) >= n_words:
            break
    return " ".join(kept_words)


def flatten_to_binary(
    df: pd.DataFrame,
    domains=("reddit_eli5",),
    length_control: bool = True,
    max_words: int = 200,
    min_words: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Flatten HC3's question/human_answers/chatgpt_answers rows into a
    binary-labeled, one-row-per-answer DataFrame.

    Only the first human answer and first chatgpt answer per question are
    kept (1:1 pairing) so classes stay balanced and no question contributes
    more than one example per label.

    If length_control=True, each pair is truncated, at sentence boundaries,
    to approximately min(len(human), len(chatgpt), max_words) words, so both
    sides of the pair are close in length and the classifier can't key on
    raw length alone. Truncation always stops at the end of a sentence
    (never mid-sentence) so it doesn't introduce a new "ends abruptly"
    shortcut in its place; as a result the two sides of a pair may differ by
    up to one sentence in word count. If length_control=False, original
    lengths are kept (aside from dropping empties/very-short answers) for
    the length-shortcut ablation.

    Returns columns: pair_id, source, question, text, label (0=human, 1=ai),
    word_count.
    """
    sub = df[df["source"].isin(domains)].reset_index(drop=True)

    records = []
    for pair_id, row in sub.iterrows():
        human_list = row["human_answers"] or []
        chatgpt_list = row["chatgpt_answers"] or []
        if not human_list or not chatgpt_list:
            continue

        human_text = _clean_text(human_list[0])
        chatgpt_text = _clean_text(chatgpt_list[0])
        if not human_text or not chatgpt_text:
            continue

        human_wc = len(human_text.split())
        chatgpt_wc = len(chatgpt_text.split())
        if human_wc < min_words or chatgpt_wc < min_words:
            continue

        if length_control:
            target_len = min(human_wc, chatgpt_wc, max_words)
            human_text = _truncate_to_words_sentence_aware(human_text, target_len)
            chatgpt_text = _truncate_to_words_sentence_aware(chatgpt_text, target_len)
            human_wc = len(human_text.split())
            chatgpt_wc = len(chatgpt_text.split())

        records.append(
            dict(
                pair_id=pair_id,
                source=row["source"],
                question=row["question"],
                text=human_text,
                label=0,
                word_count=human_wc,
            )
        )
        records.append(
            dict(
                pair_id=pair_id,
                source=row["source"],
                question=row["question"],
                text=chatgpt_text,
                label=1,
                word_count=chatgpt_wc,
            )
        )

    out = pd.DataFrame(records)
    return out.sample(frac=1, random_state=seed).reset_index(drop=True)


def group_train_val_test_split(
    df: pd.DataFrame,
    group_col: str = "pair_id",
    test_size: float = 0.15,
    val_size: float = 0.15,
    seed: int = 42,
):
    """Split into train/val/test, grouping by `group_col` so the human and
    chatgpt answer to the same question always land in the same split
    (prevents the same topic appearing on both sides of a split).
    """
    gss1 = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss1.split(df, groups=df[group_col]))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    relative_val_size = val_size / (1 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
    train_idx, val_idx = next(gss2.split(train_val_df, groups=train_val_df[group_col]))
    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    return train_df, val_df, test_df


if __name__ == "__main__":
    raw = load_hc3()
    flat = flatten_to_binary(raw, domains=("reddit_eli5",), length_control=True)
    print("flattened (length-controlled):", flat.shape)
    print(flat["label"].value_counts())
    train_df, val_df, test_df = group_train_val_test_split(flat)
    print("train/val/test:", train_df.shape, val_df.shape, test_df.shape)
