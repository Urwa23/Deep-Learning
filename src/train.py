"""
Fine-tune DistilBERT as a human-vs-AI binary text classifier on the
processed HC3 splits (see scripts/prepare_data.py).

Usage:
    python src/train.py --variant length_controlled --epochs 3
    python src/train.py --variant raw --epochs 3 --output_dir results/checkpoints/raw_ablation

`--variant raw` trains on the uncontrolled-length data, kept only to
demonstrate (via interpretability, later) that an unconstrained model
learns to exploit answer length as a shortcut.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments

from model import load_model, load_tokenizer

ROOT = Path(__file__).resolve().parent.parent


class HC3Dataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length", max_length=max_length
        )
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def load_split(variant: str, split: str, max_samples: int = None) -> pd.DataFrame:
    df = pd.read_parquet(ROOT / "data" / "processed" / variant / f"{split}.parquet")
    if max_samples is not None:
        df = df.sample(n=min(max_samples, len(df)), random_state=42).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["length_controlled", "raw"], default="length_controlled")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--max_train_samples", type=int, default=None, help="subsample for a quick smoke test")
    parser.add_argument("--max_eval_samples", type=int, default=None)
    args = parser.parse_args()

    output_dir = args.output_dir or str(ROOT / "results" / "checkpoints" / args.variant)

    train_df = load_split(args.variant, "train", args.max_train_samples)
    val_df = load_split(args.variant, "val", args.max_eval_samples)

    tokenizer = load_tokenizer()
    model = load_model()

    train_ds = HC3Dataset(train_df["text"], train_df["label"], tokenizer, args.max_length)
    val_ds = HC3Dataset(val_df["text"], val_df["label"], tokenizer, args.max_length)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"[{args.variant}] final val metrics:", metrics)

    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Saved model to {final_dir}")


if __name__ == "__main__":
    main()
