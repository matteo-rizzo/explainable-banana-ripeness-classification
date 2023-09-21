from pathlib import Path

import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from typing import Iterable


def train_val_test(target: str = "M", validation: float = .0, random_state: int = 0) -> dict[str, dict[str, list]]:
    base_dataset = Path("dataset/AMI2020")

    target = "misogynous" if target == "M" else "aggressiveness"

    # base_dataset
    train_df = pd.read_csv(base_dataset / "trainingset" / "AMI2020_training_raw_anon.tsv", sep="\t", usecols=["id", "text", target])
    test_df = pd.read_csv(base_dataset / "testset" / "AMI2020_test_raw_gold_anon.tsv", sep="\t", usecols=["id", "text", target])

    train_x, train_y, train_ids = train_df["text"].tolist(), train_df[target].tolist(), train_df["id"].tolist()
    test_x, test_y, test_ids = test_df["text"].tolist(), test_df[target].tolist(), test_df["id"].tolist()

    add_val = dict()

    if validation > 0:
        train_x, val_x, train_y, val_y, train_ids, val_ids = train_test_split(train_x, train_y, train_ids, test_size=validation, random_state=random_state,
                                                                              shuffle=True, stratify=train_y)
        add_val = {
            "val": {
                "x": val_x,
                "y": val_y,
                "ids": val_ids
            }
        }

    return {
        "train": {
            "x": train_x,
            "y": train_y,
            "ids": train_ids
        },
        "test": {
            "x": test_x,
            "y": test_y,
            "ids": test_ids
        },
        **add_val
    }


def compute_metrics(y_pred, y_true, sk_classifier_name: str = None) -> dict[str, float]:
    precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average="macro", pos_label=1)
    acc = metrics.accuracy_score(y_true, y_pred)
    if sk_classifier_name:
        print(f"{sk_classifier_name} accuracy: {acc:.3f}")
        print(f"{sk_classifier_name} precision: {precision:.3f}")
        print(f"{sk_classifier_name} recall: {recall:.3f}")
        print(f"{sk_classifier_name} F1-score: {f1_score:.3f}")

    return {"f1": f1_score, "accuracy": acc, "precision": precision, "recall": recall}


def batch_list(iterable: Iterable, batch_size: int = 10) -> Iterable:
    """
    Yields batches from an iterable container

    :param iterable: elements to be batched
    :param batch_size: (max) number of elements in a single batch
    :return: generator of batches
    """
    data_len = len(iterable)
    for ndx in range(0, data_len, batch_size):
        yield iterable[ndx:min(ndx + batch_size, data_len)]
