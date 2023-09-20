from __future__ import annotations

import os

import evaluate
import numpy as np
import torch
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer
from transformers import TrainingArguments
from transformers.trainer import Trainer

from src.cv.classifiers.deep_learning.functional.yaml_manager import load_yaml
from src.nlp.deep_learning.create_dataset import create_hf_dataset

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def compute_metrics(eval_pred):
    metric1 = evaluate.load("precision")
    metric2 = evaluate.load("recall")
    metric3 = evaluate.load("accuracy")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision = metric1.compute(predictions=predictions, references=labels)["precision"]
    recall = metric2.compute(predictions=predictions, references=labels)["recall"]
    accuracy = metric3.compute(predictions=predictions, references=labels)["accuracy"]
    return {"precision": precision, "recall": recall, "accuracy": accuracy}


if __name__ == "__main__":
    assert torch.cuda.is_available()

    config: dict = load_yaml("src/nlp/params/deep_learning.yml")
    base_model: str = config["training"]["model_name"]
    test_size: float = config["training"]["test_size"]
    model_max_length: int = config["training"]["model_max_length"]
    batch_size: int = config["training"]["batch_size"]
    freeze_base: bool = config["training"]["freeze_base"]
    epochs: bool = config["training"]["epochs"]
    resume: bool = config["training"]["resume"]

    train_ds, test_ds = create_hf_dataset()

    tokenizer = AutoTokenizer.from_pretrained(base_model)


    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=model_max_length)


    tokenized_train_ds = train_ds.map(tokenize_function, batched=True)
    tokenized_test_ds = test_ds.map(tokenize_function, batched=True)

    n_val = int(tokenized_train_ds.num_rows * test_size)

    print(f"Using {n_val} examples as validation set")
    small_train_dataset = tokenized_train_ds.shuffle(seed=42).select(range(n_val))
    small_eval_dataset = tokenized_train_ds.shuffle(seed=42).select(range(n_val, tokenized_train_ds.num_rows))

    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=2)
    if freeze_base:
        for param in model.base_model.parameters():
            param.requires_grad = False

    training_args = TrainingArguments(output_dir="dumps/nlp_models", evaluation_strategy="steps",
                                      logging_dir="dumps/nlp_models/logs", num_train_epochs=epochs,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      dataloader_num_workers=4, dataloader_pin_memory=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    checkpoint_reload: str | bool = False
    if resume:
        checkpoint_reload = config["training"]["checkpoint"]

    trainer.train(resume_from_checkpoint=checkpoint_reload)
