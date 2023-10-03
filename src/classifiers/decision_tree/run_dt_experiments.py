from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn import tree
from sklearn.model_selection import train_test_split, GridSearchCV
from torchmetrics.functional.classification import multiclass_precision, multiclass_recall, multiclass_f1_score

from src.classifiers.decision_tree.scripts.extract_rules import get_leaf_constraints
from src.classifiers.deep_learning.functional.yaml_manager import load_yaml

"""
https://scikit-learn.org/stable/modules/tree.html
"""


def prepare_features(train_config):
    # Configure paths
    dataset_folder: Path = Path("dataset")
    path_to_color_features: Path = dataset_folder / train_config["color_features"]
    path_to_bsa_features: Path = dataset_folder / train_config["bsa_features"]

    color_feature_names: List[str] = pd.read_csv(path_to_color_features, nrows=1).columns.tolist()[:-1]

    # Read color values
    y = pd.read_csv(path_to_color_features, usecols=["y"], index_col=False)

    # Configure features
    if train_config["features"]["color"] and train_config["features"]["bsa"]:
        # BSA and colors
        x_color = pd.read_csv(path_to_color_features, usecols=color_feature_names, index_col=False)
        x_bsa = pd.read_csv(path_to_bsa_features, index_col=False)
        x = pd.concat((x_color, x_bsa), axis=1)
    elif train_config["features"]["color"]:
        # Just colors
        x_color = pd.read_csv(path_to_color_features, usecols=color_feature_names, index_col=False)
        x = x_color
    elif train_config["features"]["bsa"]:
        # Just BSA
        x_bsa = pd.read_csv(path_to_bsa_features, index_col=False)
        x = x_bsa
    else:
        raise ValueError
    return x, y, color_feature_names


def grid_search_best_params():
    # Load configuration
    train_config: Dict = load_yaml("params/experiment.yml")
    num_classes: int = train_config["output_size"]
    num_rand_states: int = train_config["num_seeds"]
    test_size: float = train_config["test_size"]

    # Prepare features and target
    x, y, _ = prepare_features(train_config)
    # Initiate training
    avg_metrics: Dict[str, List] = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    grid_clf = None

    # Per-seed training
    for rs in range(num_rand_states):
        # Prepare splits
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=rs)
        y_train, y_test = y_train["y"].tolist(), y_test["y"].tolist()

        # Setup and train classifier
        grid_clf = GridSearchCV(tree.DecisionTreeClassifier(), n_jobs=-1,
                                param_grid=train_config["grid_search_params"], verbose=10, )
        grid_clf.fit(x_train, y_train)
        y_pred = grid_clf.predict(x_test).tolist()

        # Calculate metrics
        y_pred, y_test = torch.Tensor(y_pred), torch.Tensor(y_test)
        metrics = {
            "accuracy": sum([1 for a, b in zip(y_test, y_pred) if a == b]) / len(y_pred),
            "precision": float(multiclass_precision(y_pred, y_test, num_classes=num_classes)),
            "recall": float(multiclass_recall(y_pred, y_test, num_classes=num_classes)),
            "f1": float(multiclass_f1_score(y_pred, y_test, num_classes=num_classes))
        }

        # Print results
        print(f"Random Seed {rs} - Test Metrics:")
        for metric, value in metrics.items():
            print(f"\t {metric} - {''.join(['.'] * (15 - len(metric)))} : {value:.4f}")

        avg_metrics["accuracy"].append(metrics["accuracy"])
        avg_metrics["precision"].append(metrics["precision"])
        avg_metrics["recall"].append(metrics["recall"])
        avg_metrics["f1"].append(metrics["f1"])

    print("-----------------------------------------------------------")
    print(f"Average Test Metrics Over {num_rand_states} Random Seeds:")
    for metric, value in avg_metrics.items():
        print(f"\t {metric} - {''.join(['.'] * (15 - len(metric)))} : {np.mean(value):.4f} ({np.std(value):.4f})")

    print("-----------------------------------------------------------")
    print(grid_clf.best_params_)


def train_dt():
    # Load configuration
    train_config: Dict = load_yaml("params/experiment.yml")
    num_classes: int = train_config["output_size"]
    num_rand_states: int = train_config["num_seeds"]
    test_size: float = train_config["test_size"]

    # Prepare features and target
    x, y, color_feature_names = prepare_features(train_config)
    # Initiate training
    avg_metrics: Dict[str, List] = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    decision_tree = None

    # Per-seed training
    for rs in range(num_rand_states):
        # Prepare splits
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=rs)
        y_train, y_test = y_train["y"].tolist(), y_test["y"].tolist()

        # Setup and train classifier
        decision_tree = tree.DecisionTreeClassifier(random_state=rs, **train_config["tree_params"])
        decision_tree.fit(x_train, y_train)
        y_pred = decision_tree.predict(x_test).tolist()

        # Calculate metrics
        y_pred, y_test = torch.Tensor(y_pred), torch.Tensor(y_test)
        metrics = {
            "accuracy": sum([1 for a, b in zip(y_test, y_pred) if a == b]) / len(y_pred),
            "precision": float(multiclass_precision(y_pred, y_test, num_classes=num_classes)),
            "recall": float(multiclass_recall(y_pred, y_test, num_classes=num_classes)),
            "f1": float(multiclass_f1_score(y_pred, y_test, num_classes=num_classes))
        }

        # Print results
        print(f"Random Seed {rs} - Test Metrics:")
        for metric, value in metrics.items():
            print(f"\t {metric} - {''.join(['.'] * (15 - len(metric)))} : {value:.4f}")

        avg_metrics["accuracy"].append(metrics["accuracy"])
        avg_metrics["precision"].append(metrics["precision"])
        avg_metrics["recall"].append(metrics["recall"])
        avg_metrics["f1"].append(metrics["f1"])

    print("-----------------------------------------------------------")
    print(f"Average Test Metrics Over {num_rand_states} Random Seeds:")
    for metric, value in avg_metrics.items():
        print(f"\t {metric} - {''.join(['.'] * (15 - len(metric)))} : {np.mean(value):.4f} ({np.std(value):.4f})")

    return decision_tree, color_feature_names, num_classes


if __name__ == "__main__":
    grid_search_best_params()
    dt, f, c = train_dt()
    leaves = get_leaf_constraints(dt, f, c)
