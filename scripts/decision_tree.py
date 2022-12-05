import os

import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
from torchmetrics.functional.classification import multiclass_precision, multiclass_recall, multiclass_f1_score

"""
https://scikit-learn.org/stable/modules/tree.html
"""


def main():
    num_classes = 4

    path_to_data = os.path.join("dataset")
    dataset = pd.read_csv(path_to_data)

    x, y = dataset["x"], dataset["y"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

    decision_tree = tree.DecisionTreeClassifier()
    decision_tree.fit(x_train, y_train)
    y_pred = decision_tree.predict(x_test)

    metrics = {
        "precision": float(multiclass_precision(y_pred, y_test, num_classes=num_classes)),
        "recall": float(multiclass_recall(y_pred, y_test, num_classes=num_classes)),
        "macro_f1": float(multiclass_f1_score(y_pred, y_test, num_classes=num_classes))
    }

    print("Test metrics:")
    for metric, value in metrics.items():
        print(f"\t {metric} - {''.join(['.'] * (15 - len(metric)))} : {value:.4f}")

    tree.plot_tree(decision_tree)


if __name__ == '__main__':
    main()
