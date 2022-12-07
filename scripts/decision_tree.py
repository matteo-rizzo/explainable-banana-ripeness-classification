import os

import numpy as np
import pandas as pd
import torch
from sklearn import tree
from sklearn.model_selection import train_test_split
from torchmetrics.functional.classification import multiclass_precision, multiclass_recall, multiclass_f1_score

"""
https://scikit-learn.org/stable/modules/tree.html
"""


def main():
    num_classes = 4
    num_rand_states = 10

    path_to_data = os.path.join("dataset", "treviso-market-224_224-avg_col-seg-fill_holes.csv")
    x = pd.read_csv(path_to_data, usecols=["r", "g", "b"], index_col=False)
    y = pd.read_csv(path_to_data, usecols=["y"], index_col=False)

    avg_metrics = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for rs in range(10):

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=rs)
        y_train, y_test = y_train["y"].tolist(), y_test["y"].tolist()

        decision_tree = tree.DecisionTreeClassifier(random_state=rs)
        decision_tree.fit(x_train, y_train)
        y_pred = decision_tree.predict(x_test).tolist()

        y_pred, y_test = torch.Tensor(y_pred), torch.Tensor(y_test)

        metrics = {
            "accuracy": sum([1 for a, b in zip(y_test, y_pred) if a == b]) / len(y_pred),
            "precision": float(multiclass_precision(y_pred, y_test, num_classes=num_classes)),
            "recall": float(multiclass_recall(y_pred, y_test, num_classes=num_classes)),
            "f1": float(multiclass_f1_score(y_pred, y_test, num_classes=num_classes))
        }

        print(f"Random Seed {rs} - Test Metrics:")
        for metric, value in metrics.items():
            print(f"\t {metric} - {''.join(['.'] * (15 - len(metric)))} : {value:.4f}")

        avg_metrics["accuracy"].append(metrics["accuracy"])
        avg_metrics["precision"].append(metrics["precision"])
        avg_metrics["recall"].append(metrics["recall"])
        avg_metrics["f1"].append(metrics["f1"])

    print("\n-----------------------------------------------------------\n")
    print(f"\nAverage Test Metrics Over {num_rand_states} Random Seeds:")
    for metric, value in avg_metrics.items():
        print(f"\t {metric} - {''.join(['.'] * (15 - len(metric)))} : {np.mean(value):.4f} ({np.std(value):.4f})")

    # tree.plot_tree(decision_tree,
    #                feature_names=["r", "g", "b"],
    #                class_names=["0", "1", "2", "3"],
    #                label="none",
    #                filled=True)
    # fig = gcf()
    # fig.set_size_inches(5, 5, forward=True)
    # fig.set_dpi(1000)
    # plt.show()


if __name__ == '__main__':
    main()
