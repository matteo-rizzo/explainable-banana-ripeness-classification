import copy
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from matplotlib.pyplot import gcf
from sklearn import tree
from sklearn.model_selection import train_test_split
from torchmetrics.functional.classification import multiclass_precision, multiclass_recall, multiclass_f1_score

from functional.yaml_manager import load_yaml

"""
https://scikit-learn.org/stable/modules/tree.html
"""


def train_dt():
    # Load configuration
    train_config: Dict = load_yaml("params/networks/decision_tree.yml")
    num_classes: int = train_config["output_size"]
    num_rand_states: int = train_config["num_seeds"]

    # Configure paths
    dataset_folder: Path = Path("dataset")
    path_to_color_features: Path = dataset_folder / train_config["color_features"]
    path_to_bsa_features: Path = dataset_folder / train_config["bsa_features"]
    color_feature_names: List[str] = pd.read_csv(path_to_color_features, nrows=1).columns.tolist()[:-1]

    # Read values
    x_color = pd.read_csv(path_to_color_features, usecols=color_feature_names, index_col=False)
    x_bsa = pd.read_csv(path_to_bsa_features, index_col=False)
    y = pd.read_csv(path_to_color_features, usecols=["y"], index_col=False)

    if train_config["features"]["color"] and train_config["features"]["bsa"]:
        x = pd.concat((x_color, x_bsa), axis=1)
    elif train_config["features"]["color"]:
        x = x_color
    elif train_config["features"]["bsa"]:
        x = x_bsa
    else:
        raise ValueError
    # Initiate training
    avg_metrics: Dict[str, List] = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    # Per-seed training
    for rs in range(num_rand_states):
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

    return decision_tree, color_feature_names, num_classes


def merge_rules(per_class_rules: List[List[List[Tuple[float, float]]]], new_rules: Dict[str, List[Tuple[float, bool]]], class_idx: int, max_val: float, min_val: float) -> None:
    to_update: List[List[Tuple[float, float]]] = per_class_rules[class_idx]

    feature_ranges: List[Tuple[float, float]] = list()
    for feature_name, path in new_rules.items():
        node_min = min_val
        node_max = max_val
        for (node_threshold, sign) in path:
            if sign is False:
                # <=
                if node_threshold < node_max:
                    node_max = node_threshold
            else:
                # >
                if node_threshold > node_min:
                    node_min = node_threshold

            feature_ranges.append((node_min, node_max))
        if not path:
            feature_ranges.append((node_min, node_max))
    to_update.append(feature_ranges)


def get_leaf_constraints(clf: tree.DecisionTreeClassifier, feature_names: List[Any], num_classes: int) -> List[List[List[Tuple[float, float]]]]:
    """
    For each target class, return the set of constraints on each feature that lead to that classification

    :param clf: decision tree classifier (DT)
    :param num_classes: number of predicted classes in the DT
    :param feature_names: list of names to use for each feature in the DT, with their position matching the position in the DT
    :return: for each output class, a list of feature ranges that lead to that prediction (as tuple of feature values)
    """
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    prediction = clf.tree_.value.argmax(axis=2).reshape(-1)

    # For each class, we want a list of leaf rules leading to that class
    # Each leaf rule is a List of ranges (min-max) for each feature
    per_class_rules: List[List[List[Tuple[float, float]]]] = [[] for _ in range(num_classes)]
    rules: Dict[str, List[Tuple[float, bool]]] = {fn: list() for fn in feature_names}

    stack = [(0, rules)]  # start with the root node id (0) and its depth (0)
    while len(stack) > 0:
        # `pop` ensures each node is only visited once
        node_id, rule_path = stack.pop()

        node_feature, node_threshold = feature[node_id], threshold[node_id]

        is_split_node = children_left[node_id] != children_right[node_id]
        if is_split_node:
            # Decision node

            ln = copy.deepcopy(rule_path)
            rn = copy.deepcopy(rule_path)

            ln[feature_names[node_feature]].append((node_threshold, False))  # <=
            rn[feature_names[node_feature]].append((node_threshold, True))  # >

            # Append left and right children and depth to `stack`
            stack.append((children_left[node_id], ln))
            stack.append((children_right[node_id], rn))
        else:
            # Leaf node
            if feature_names == ["u", "v"]:
                # U and V values should be in [-.5, .5]
                max_val = .5
                min_val = -.5
            else:
                # RGB in [0, 1]
                max_val = 1.0
                min_val = .0
            merge_rules(per_class_rules, rule_path, prediction[node_id], max_val=max_val, min_val=min_val)

    assert sum([len(a) for a in per_class_rules]) == clf.tree_.n_leaves, "Wrong number of leaves and outputs"
    return per_class_rules


if __name__ == "__main__":
    dt, f, c = train_dt()
    leaves = get_leaf_constraints(dt, f, c)
    pass
