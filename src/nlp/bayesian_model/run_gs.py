from pprint import pprint
from typing import Type

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.model_selection import GridSearchCV

from src.cv.classifiers.deep_learning.functional.yaml_manager import load_yaml
from src.nlp.bayesian_model.bayesian_pipeline import fit_pipeline
from src.nlp.bayesian_model.bayesian_ridge_classifier import BayesianRidgeClassifier
from src.nlp.dataset import train_val_test, compute_metrics


def grid_search_best_params(sk_classifier_type: Type[ClassifierMixin], target: str = "M"):
    # Load configuration
    train_config: dict = load_yaml("src/nlp/params/experiment.yml")
    num_rand_states: int = train_config["grid_search_params"]["num_seeds"]
    test_size: float = train_config["grid_search_params"]["test_size"]

    # Initiate training
    avg_metrics: dict[str, list] = {"accuracy": [], "precision": [], "recall": [], "f1": []}
    best_params = list()

    # Per seed training
    for rs in range(num_rand_states):
        # Prepare splits
        val_data = train_val_test(target=target, random_state=rs, validation=test_size)

        # Setup and train classifier
        params = train_config["grid_search_params"][sk_classifier_type.__name__]

        gs = GridSearchCV(sk_classifier_type(), param_grid=params, verbose=10, refit=True)
        grid_clf = make_pipeline(gs)

        grid_clf.fit(val_data["train"]["x"], val_data["train"]["y"])
        y_pred = grid_clf.predict(val_data["val"]["x"]).tolist()

        # Calculate metrics
        metrics = compute_metrics(y_pred, val_data["val"]["y"])

        # Print results
        print(f"Random Seed {rs} - Validation Metrics:")
        for metric, value in metrics.items():
            print(f"\t {metric} - {''.join(['.'] * (15 - len(metric)))} : {value:.4f}")

        avg_metrics["accuracy"].append(metrics["accuracy"])
        avg_metrics["precision"].append(metrics["precision"])
        avg_metrics["recall"].append(metrics["recall"])
        avg_metrics["f1"].append(metrics["f1"])

        best_params.append(gs.best_params_)

    print("-----------------------------------------------------------")
    print(f"Average Validation Metrics Over {num_rand_states} Random Seeds:")
    for metric, value in avg_metrics.items():
        print(f"\t {metric} - {''.join(['.'] * (15 - len(metric)))} : {np.mean(value):.4f} ({np.std(value):.4f})")

    print("-----------------------------------------------------------")
    pprint(best_params)


if __name__ == "__main__":
    print("*** GRID SEARCH ")
    grid_search_best_params(BayesianRidgeClassifier, target="M")
