from pathlib import Path
from pprint import pprint
from typing import Type

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline

from src.nlp.scripts.text_features import TextFeatureExtractor
from src.cv.classifiers.deep_learning.functional.yaml_manager import load_yaml


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
        train_x, val_x, train_y, val_y, train_ids, val_ids = train_test_split(train_x, train_y, train_ids, test_size=validation, random_state=random_state)
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
        print(f"{sk_classifier_name} accuracy:", acc)
        print(f"{sk_classifier_name} precision:", precision)
        print(f"{sk_classifier_name} recall:", recall)
        print(f"{sk_classifier_name} F1-score:", f1_score)

    return {"f1": f1_score, "accuracy": acc, "precision": precision, "recall": recall}


def make_pipeline(sk_classifier: ClassifierMixin) -> Pipeline:
    fex = TextFeatureExtractor()
    bow_vectorizer = TfidfVectorizer(tokenizer=fex.preprocessing_tokenizer,
                                     ngram_range=(1, 3),
                                     max_features=10000,
                                     token_pattern=None)

    # Create a pipeline using TF-IDF
    pipe = Pipeline([('vectorizer', bow_vectorizer),
                     ('classifier', sk_classifier)])
    return pipe


def naive_classifier(sk_classifier: ClassifierMixin, training_data: dict[str, dict[str, list]],
                     return_pipe: bool = False) -> np.ndarray | tuple[np.ndarray, Pipeline]:
    pipe = make_pipeline(sk_classifier)

    print("------ Training")

    pipe.fit(training_data["train"]["x"], training_data["train"]["y"])

    print("------ Testing")

    # Predicting with a test dataset
    predicted = pipe.predict(training_data["test"]["x"])

    if not return_pipe:
        return predicted
    else:
        return predicted, pipe


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
    grid_search_best_params(RidgeClassifier, target="M")
