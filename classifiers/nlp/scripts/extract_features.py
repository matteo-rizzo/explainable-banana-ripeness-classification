import zipfile
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

from classifiers.nlp.scripts.text_features import TextFeatureExtractor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
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


def naive_classifier(data: dict[str, dict[str, list]]):
    classifier = LogisticRegression()

    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.25)

    fex = TextFeatureExtractor()
    bow_vectorizer = TfidfVectorizer(tokenizer=fex.preprocessing_tokenizer, ngram_range=(1, 3), max_features=10000)

    # Create a pipeline using TF-IDF
    pipe = Pipeline([('vectorizer', bow_vectorizer),
                     ('classifier', classifier)])

    print("Training")

    pipe.fit(data["train"]["x"], data["train"]["y"])

    print("Testing")

    # Predicting with a test dataset
    predicted = pipe.predict(data["test"]["x"])

    # Model Accuracy
    classifier_name = classifier.__class__.__name__
    precision, recall, f1_score, _ = metrics.precision_recall_fscore_support(data["test"]["y"], predicted, average="binary", pos_label=1)
    print(f"{classifier_name} accuracy:", metrics.accuracy_score(data["test"]["y"], predicted))
    print(f"{classifier_name} precision:", precision)
    print(f"{classifier_name} recall:", recall)
    print(f"{classifier_name} F1-score:", f1_score)

    return predicted


if __name__ == '__main__':
    # Read data
    data = train_val_test(target="M")
    m_pred = naive_classifier(data)

    data = train_val_test(target="A")
    a_pred = naive_classifier(data)
    # a_pred = [0] * len(m_pred)

    f = "results/bestTeam.A.r.u.run1"
    pd.DataFrame([m_pred, a_pred], columns=data["test"]["ids"]).T.to_csv(f, header=False, sep="\t")

    # create a ZipFile object in write mode
    with zipfile.ZipFile(f"{f}.zip", "w", zipfile.ZIP_DEFLATED) as zipf:
        # add the file to the zip file
        zipf.write(f)

    # Best task score: 0.707
