from __future__ import annotations

import os
import re
from pathlib import Path

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


# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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


def get_next_run_name(base_path: str | Path, model_name: str) -> Path:
    """
    Search correct sequence for checkpoint folders and return new name
    This is useful not to overwrite existing dumps.
    """
    base_path = Path(base_path)
    # Get a list of all items in the directory
    items = os.listdir(base_path)

    # Filter out items that match the pattern "path_{N}"
    matches = [item for item in items if re.fullmatch(rf"{model_name}_\d+", item)]

    if not matches:
        # If there are no matches, return "path_1"
        max_num = 0
    else:
        # If there are matches, extract the maximum number X and return "path_{X+1}"
        max_num = max(int(re.search(r"\d+", match).group()) for match in matches)
    return base_path / f"{model_name}_{max_num + 1}"


if __name__ == "__main__":
    assert torch.cuda.is_available()

    config: dict = load_yaml("src/nlp/params/deep_learning.yml")
    base_model: str = config["training"]["model_name"]
    eval_size: float = config["training"]["eval_size"]
    model_max_length: int = config["training"]["model_max_length"]
    batch_size: int = config["training"]["batch_size"]
    freeze_base: bool = config["training"]["freeze_base"]
    epochs: bool = config["training"]["epochs"]
    resume: bool = config["training"]["resume"]
    lr: float = config["training"].get("learning_rate", 5.0e-5)
    wd: float = config["training"].get("decay", 0.0)
    use_gpu: bool = config["use_gpu"]

    base_model_path: Path = Path("dumps") / "nlp_models"
    base_model_path.mkdir(parents=True, exist_ok=True)

    if resume:
        checkpoint_model_path = config["training"]["checkpoint"]
    else:
        checkpoint_model_path = get_next_run_name(base_model_path, base_model.replace("/", "_"))

    train_ds, test_ds = create_hf_dataset(target="M")

    tokenizer = AutoTokenizer.from_pretrained(base_model)


    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=model_max_length)


    tokenized_train_ds = train_ds.map(tokenize_function, batched=True)
    tokenized_test_ds = test_ds.map(tokenize_function, batched=True)

    print(f"Using {int(tokenized_train_ds.num_rows * eval_size)} examples as validation set")
    # small_train_dataset = tokenized_train_ds.shuffle(seed=42).select(range(n_val))
    # small_eval_dataset = tokenized_train_ds.shuffle(seed=42).select(range(n_val, tokenized_train_ds.num_rows))
    train_eval_dataset = tokenized_train_ds.train_test_split(test_size=eval_size, shuffle=True, seed=39, stratify_by_column="label")

    model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=train_eval_dataset["train"].features["label"].num_classes)
    if freeze_base:
        for param in model.base_model.parameters():
            param.requires_grad = False

    optim = "adamw_torch"  # "adamw_torch_fused" if use_gpu else "adamw_torch"
    training_args = TrainingArguments(output_dir=checkpoint_model_path,
                                      overwrite_output_dir=resume,
                                      num_train_epochs=epochs,
                                      per_device_train_batch_size=batch_size,
                                      per_device_eval_batch_size=batch_size,
                                      dataloader_num_workers=4, dataloader_pin_memory=True,
                                      optim=optim, learning_rate=lr, weight_decay=wd,
                                      use_cpu=not use_gpu, seed=943, data_seed=3211,
                                      save_strategy="epoch", evaluation_strategy="epoch", logging_strategy="epoch",
                                      metric_for_best_model="eval_loss", save_total_limit=3, load_best_model_at_end=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_eval_dataset["train"],
        eval_dataset=train_eval_dataset["test"],  # validation set
        compute_metrics=compute_metrics,
        tokenizer=tokenizer  # this is needed to load correctly for inference
    )

    trainer.train(resume_from_checkpoint=resume)
