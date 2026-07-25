"""DistilBERT binary sequence-classification model + tokenizer helpers."""
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

MODEL_NAME = "distilbert-base-uncased"


def load_tokenizer(model_name: str = MODEL_NAME) -> DistilBertTokenizerFast:
    return DistilBertTokenizerFast.from_pretrained(model_name)


def load_model(model_name: str = MODEL_NAME, num_labels: int = 2) -> DistilBertForSequenceClassification:
    return DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
