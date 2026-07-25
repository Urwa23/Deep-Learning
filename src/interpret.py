"""
Captum-based interpretability for the fine-tuned DistilBERT classifier.

Provides word-level attributions via:
  - Saliency (gradient magnitude w.r.t. input embeddings)
  - Integrated Gradients (path-integrated gradient attribution, embedding baseline = [PAD])

Both explain the predicted class's logit as a function of the model's input
embeddings, then sum each token's embedding-dimension attributions into a
single per-token score.
"""
from pathlib import Path

import torch
from captum.attr import IntegratedGradients, Saliency

ROOT = Path(__file__).resolve().parent.parent


class DistilBertEmbeddingWrapper(torch.nn.Module):
    """Wraps a DistilBERT classifier so Captum can attribute w.r.t. input embeddings
    instead of token ids (which aren't differentiable)."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs_embeds, attention_mask):
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return out.logits


def _embed(model, input_ids):
    return model.distilbert.embeddings(input_ids)


def get_word_attributions(model, tokenizer, text: str, target: int = None, method: str = "ig", max_length: int = 256, n_steps: int = 50):
    """Return (tokens, attributions, predicted_label, predicted_prob) for a single text.

    method: "ig" (Integrated Gradients) or "saliency".
    target: class index to attribute w.r.t.; defaults to the model's predicted class.
    """
    model.eval()
    device = next(model.parameters()).device

    enc = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs = torch.softmax(logits, dim=-1)
        pred_label = int(torch.argmax(probs, dim=-1).item())
        pred_prob = float(probs[0, pred_label].item())

    if target is None:
        target = pred_label

    wrapper = DistilBertEmbeddingWrapper(model)
    input_embeds = _embed(model, input_ids)

    if method == "saliency":
        attributor = Saliency(wrapper)
        attributions = attributor.attribute(
            input_embeds, target=target, additional_forward_args=(attention_mask,)
        )
    elif method == "ig":
        pad_id = tokenizer.pad_token_id
        baseline_ids = torch.full_like(input_ids, pad_id)
        baseline_embeds = _embed(model, baseline_ids)
        attributor = IntegratedGradients(wrapper)
        attributions = attributor.attribute(
            input_embeds,
            baselines=baseline_embeds,
            target=target,
            additional_forward_args=(attention_mask,),
            n_steps=n_steps,
        )
    else:
        raise ValueError(f"unknown method: {method}")

    # sum attribution over the embedding dimension -> one score per token
    token_scores = attributions.sum(dim=-1).squeeze(0).detach().cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).cpu().tolist())

    return tokens, token_scores, pred_label, pred_prob


def top_k_tokens(tokens, scores, k: int = 10):
    """Return the top-k tokens by |attribution|, excluding special tokens."""
    special = {"[CLS]", "[SEP]", "[PAD]"}
    pairs = [(t, s) for t, s in zip(tokens, scores) if t not in special]
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return pairs[:k]


if __name__ == "__main__":
    import argparse

    from model import load_tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="path to a trainer.save_model() dir")
    parser.add_argument("--text", required=True)
    parser.add_argument("--method", choices=["ig", "saliency"], default="ig")
    args = parser.parse_args()

    from transformers import DistilBertForSequenceClassification

    tok = load_tokenizer(args.checkpoint)
    mdl = DistilBertForSequenceClassification.from_pretrained(args.checkpoint)

    tokens, scores, pred, prob = get_word_attributions(mdl, tok, args.text, method=args.method)
    print(f"predicted label: {pred} (p={prob:.3f})  [0=human, 1=ai]")
    for tok_str, score in top_k_tokens(tokens, scores):
        print(f"{score:+.4f}  {tok_str}")
