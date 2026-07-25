"""
Day-1 exploration of the HC3 dataset: domain sizes, a manual sample read-through,
and human-vs-chatgpt length distributions (to check for a length-shortcut risk).
"""
from huggingface_hub import hf_hub_download
import pandas as pd
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

pd.set_option("display.width", 120)
pd.set_option("display.max_colwidth", 80)

path = hf_hub_download(
    repo_id="Hello-SimpleAI/HC3",
    filename="all.jsonl",
    repo_type="dataset",
)

rows = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)
print("=== shape ===")
print(df.shape)

print("\n=== domain (source) value counts ===")
print(df["source"].value_counts())

# --- Manual sample read-through: a few triples per domain ---
print("\n=== sample triples (2 per domain) ===")
for src in df["source"].unique():
    sub = df[df["source"] == src].sample(n=min(2, len(df[df["source"] == src])), random_state=42)
    for _, row in sub.iterrows():
        print(f"\n--- domain: {src} ---")
        print("Q:", row["question"][:200])
        h = row["human_answers"][0] if row["human_answers"] else ""
        c = row["chatgpt_answers"][0] if row["chatgpt_answers"] else ""
        print("HUMAN:", h[:300])
        print("CHATGPT:", c[:300])

# --- Word-count distribution: human vs chatgpt, overall and per domain ---
def word_counts(answer_list):
    return [len(a.split()) for a in answer_list] if isinstance(answer_list, list) else []

df["human_wc"] = df["human_answers"].apply(lambda lst: word_counts(lst))
df["chatgpt_wc"] = df["chatgpt_answers"].apply(lambda lst: word_counts(lst))

human_wc_flat = [wc for lst in df["human_wc"] for wc in lst]
chatgpt_wc_flat = [wc for lst in df["chatgpt_wc"] for wc in lst]

human_wc_s = pd.Series(human_wc_flat)
chatgpt_wc_s = pd.Series(chatgpt_wc_flat)

print("\n=== overall word-count distribution ===")
print("HUMAN   :\n", human_wc_s.describe())
print("\nCHATGPT :\n", chatgpt_wc_s.describe())

print("\n=== per-domain median word count (human vs chatgpt) ===")
per_domain = []
for src in df["source"].unique():
    sub = df[df["source"] == src]
    h = pd.Series([wc for lst in sub["human_wc"] for wc in lst])
    c = pd.Series([wc for lst in sub["chatgpt_wc"] for wc in lst])
    per_domain.append({
        "domain": src,
        "n_rows": len(sub),
        "human_median_wc": h.median(),
        "chatgpt_median_wc": c.median(),
        "human_mean_wc": h.mean(),
        "chatgpt_mean_wc": c.mean(),
    })
print(pd.DataFrame(per_domain).to_string(index=False))
