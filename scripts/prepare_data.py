"""
Build the processed train/val/test splits and write them to data/processed/.

Produces two variants:
  - length_controlled: human/chatgpt pairs truncated to equal word count
    (the main dataset used for training/evaluation)
  - raw: original, uncontrolled lengths (used only for the length-shortcut
    ablation in the interpretability stage)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data_utils import load_hc3, flatten_to_binary, group_train_val_test_split

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
DOMAINS = ("reddit_eli5",)


def build_and_save(raw_df, length_control: bool, tag: str):
    flat = flatten_to_binary(raw_df, domains=DOMAINS, length_control=length_control)
    train_df, val_df, test_df = group_train_val_test_split(flat)

    out_dir = PROCESSED_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(out_dir / "train.parquet", index=False)
    val_df.to_parquet(out_dir / "val.parquet", index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)

    print(f"[{tag}] total={flat.shape[0]} train={train_df.shape[0]} "
          f"val={val_df.shape[0]} test={test_df.shape[0]}")
    print(f"[{tag}] label balance (train): {train_df['label'].value_counts().to_dict()}")


if __name__ == "__main__":
    raw = load_hc3()
    build_and_save(raw, length_control=True, tag="length_controlled")
    build_and_save(raw, length_control=False, tag="raw")
