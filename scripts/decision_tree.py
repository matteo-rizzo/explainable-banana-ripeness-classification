import copy
import os
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from sklearn import tree
from sklearn.model_selection import train_test_split
from torchmetrics.functional.classification import multiclass_precision, multiclass_recall, multiclass_f1_score

"""
https://scikit-learn.org/stable/modules/tree.html
"""

FEATURES_FILE = "treviso-market-224_224-hull-seg-YUV.csv"


def main():
    num_classes = 4
    num_rand_states = 10
    path_to_data = os.path.join("dataset", FEATURES_FILE)
    feature_names = pd.read_csv(path_to_data, nrows=1).columns.tolist()[:-1]

    x = pd.read_csv(path_to_data, usecols=feature_names, index_col=False)
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

    # Explain the last decision tree
    rules_extracted = get_leaf_constraints(decision_tree, feature_names=feature_names, num_classes=num_classes)
    print(rules_extracted)


def merge_rules(per_class_rules: List[List[Tuple[float]]], new_rules: Dict[str, List[float]], class_idx: int, max_val: float):
    to_update: List[Tuple[float]] = per_class_rules[class_idx]

    color_tuple = tuple([min(new_rules[k], default=max_val) for k in new_rules.keys()])
    to_update.append(color_tuple)


def get_leaf_constraints(clf: tree.DecisionTreeClassifier, feature_names: List[Any], num_classes: int):
    """
    For each target class, return the set of constraints on each feature that lead to that classification

    :param clf: po
    :return: List
    """
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    prediction = clf.tree_.value.argmax(axis=2).reshape(-1)

    per_class_rules: List[List[Tuple[float, float, float]]] = [[] for _ in range(num_classes)]
    rules: Dict[str, List[float]] = {fn: list() for fn in feature_names}

    stack = [(0, rules)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, rule_path = stack.pop()

        f, t = feature[node_id], threshold[node_id]

        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            # Decision node
            rule_path[feature_names[f]].append(t)

            # Append left and right children and depth to `stack`
            stack.append((children_right[node_id], copy.deepcopy(rule_path), f))
            stack.append((children_left[node_id], copy.deepcopy(rule_path), f))
        else:
            # Leaf node
            if feature_names == ["u", "v"]:
                # U and V values should be in [-.5, .5]
                max_val = .5
            else:
                max_val = 1.0
            merge_rules(per_class_rules, rule_path, prediction[node_id], max_val=max_val)

    assert sum([len(a) for a in per_class_rules]) == clf.tree_.n_leaves, "Wrong number of leaves and outputs"
    return per_class_rules


if __name__ == "__main__":
    main()
