from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchmetrics.functional.classification import multiclass_precision, multiclass_recall, multiclass_f1_score


def evaluate(file_name: Path):
    df = pd.read_csv(file_name, sep=",", index_col=False, header=0, dtype=np.int)

    gt = torch.Tensor(df["gt"].tolist())
    names_sorted = ["mm", "az", "mr"]

    assert df.min(axis=0).max() == 0, f"Error, wrong min class value: {df.min()}"
    assert df.max(axis=0).min() == 3, f"Error, wrong min class value: {df.max()}"

    num_classes = 4

    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": []
    }

    for name in names_sorted:
        y_pred = torch.Tensor(df[name].tolist())

        metrics["accuracy"].append(sum([1 for a, b in zip(gt, y_pred) if a == b]) / len(y_pred))
        metrics["precision"].append(float(multiclass_precision(y_pred, gt, num_classes=num_classes)))
        metrics["recall"].append(float(multiclass_recall(y_pred, gt, num_classes=num_classes)))
        metrics["f1"].append(float(multiclass_f1_score(y_pred, gt, num_classes=num_classes)))

    res = pd.DataFrame.from_dict(metrics, orient="index", columns=names_sorted)

    res["avg"] = res.mean(axis=1)
    res["std"] = res.std(axis=1)
    res = res.round(4)

    print(res)

    res.to_csv(Path("human_evaluation") / "results.csv", sep=",")


if __name__ == "__main__":
    file = Path("human_evaluation") / "input.csv"
    evaluate(file)
