import zipfile
from operator import itemgetter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from classifiers.nlp.scripts.text_features import TextFeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics


def train_val_test(target: str = "M", validation: float = .0) -> dict[str, dict[str, list]]:
    base_dataset = Path("dataset/AMI2020")

    target = "misogynous" if target == "M" else "aggressiveness"

    # base_dataset
    train_df = pd.read_csv(base_dataset / "trainingset" / "AMI2020_training_raw_anon.tsv", sep="\t", usecols=["id", "text", target])
    test_df = pd.read_csv(base_dataset / "testset" / "AMI2020_test_raw_gold_anon.tsv", sep="\t", usecols=["id", "text", target])

    train_x, train_y, train_ids = train_df["text"].tolist(), train_df[target].tolist(), train_df["id"].tolist()
    test_x, test_y, test_ids = test_df["text"].tolist(), test_df[target].tolist(), test_df["id"].tolist()

    add_val = dict()

    if validation > 0:
        train_x, val_x, train_y, val_y, train_ids, val_ids = train_test_split(train_x, train_y, train_ids, test_size=validation)
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


def compute_metrics(sk_classifier: ClassifierMixin, y_pred, y_true) -> float:
    classifier_name = sk_classifier.__class__.__name__
    precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(y_true, y_pred, average="macro", pos_label=1)
    print(f"{classifier_name} accuracy:", metrics.accuracy_score(y_true, y_pred))
    print(f"{classifier_name} precision:", precision)
    print(f"{classifier_name} recall:", recall)
    print(f"{classifier_name} F1-score:", f1_score)

    return f1_score


def naive_classifier(sk_classifier: ClassifierMixin, training_data: dict[str, dict[str, list]]) -> np.ndarray:
    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)

    fex = TextFeatureExtractor()
    bow_vectorizer = TfidfVectorizer(tokenizer=fex.preprocessing_tokenizer, ngram_range=(1, 3), max_features=10000, token_pattern=None)

    # Create a pipeline using TF-IDF
    pipe = Pipeline([('vectorizer', bow_vectorizer),
                     ('classifier', sk_classifier)])

    print("------ Training")

    pipe.fit(training_data["train"]["x"], training_data["train"]["y"])

    print("------ Testing")

    # Predicting with a test dataset
    predicted = pipe.predict(training_data["test"]["x"])

    return predicted


if __name__ == "__main__":
    classifier = RidgeClassifier()
    # Read data
    print("*** Misogyny task")
    data = train_val_test(target="M")
    m_pred = naive_classifier(classifier, data)
    m_f1 = compute_metrics(classifier, m_pred, data["test"]["y"])

    # Get rows with predicted misogyny
    misogyny_indexes, misogyny_ids = zip(*[(i, pid) for i, (p, pid) in enumerate(zip(m_pred, data["test"]["ids"])) if p > 0])
    non_misogyny_ids: set[int] = set(data["test"]["ids"]) - set(misogyny_ids)

    print("*** Aggressiveness task")
    data = train_val_test(target="A")

    # data_aggressiveness = {
    #     k: {
    #         "x": list(itemgetter(*misogyny_indexes)(v["x"])),
    #         "y": list(itemgetter(*misogyny_indexes)(v["y"])),
    #         "ids": list(itemgetter(*misogyny_indexes)(v["ids"]))
    #     } for k, v in data.items()
    # }

    classifier = RidgeClassifier()
    a_pred = naive_classifier(classifier, data)
    a_true = data["test"]["y"]
    a_ids = data["test"]["ids"]
    # a_pred = [0] * len(m_pred)

    # a_pred = np.concatenate([a_pred, np.array([0] * len(non_misogyny_ids))])
    # a_ids = data_aggressiveness["test"]["ids"] + list(non_misogyny_ids)
    # a_true = data_aggressiveness["test"]["y"] + ([0] * len(non_misogyny_ids))

    a_f1 = compute_metrics(classifier, a_pred, a_true)

    a_score = (m_f1 + a_f1) / 2
    print(f"\n*** Task A score: {a_score:.5f} ***")

    f = "results/bestTeam.A.r.u.run1"
    Path(f).parent.mkdir(exist_ok=True, parents=True)
    pd.DataFrame([m_pred, a_pred], columns=a_ids).T.to_csv(f, header=False, sep="\t")

    # create a ZipFile object in write mode
    with zipfile.ZipFile(f"{f}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        # add the file to the zip file
        zipf.write(f)

    # Best task A score: 0.707
