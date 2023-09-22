# Use a pipeline as a high-level helper
from __future__ import annotations

from transformers import pipeline


# Load model directly
# from transformers import AutoTokenizer, AutoModelForSequenceClassification


def create_hf_pipeline(model_name: str, device: int | str, batch_size: int = None) -> pipeline:
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)

    pipe = pipeline("text-classification", model=model_name, device=device, batch_size=batch_size)
    return pipe
