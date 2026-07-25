from huggingface_hub import hf_hub_download
import pandas as pd
import json

path = hf_hub_download(
    repo_id="Hello-SimpleAI/HC3",
    filename="all.jsonl",
    repo_type="dataset"   # <-- this was missing
)

rows = []
with open(path, "r", encoding="utf-8") as f:
    for line in f:
        rows.append(json.loads(line))

df = pd.DataFrame(rows)
print(df.shape)
print(df.columns)
print(df.iloc[0])