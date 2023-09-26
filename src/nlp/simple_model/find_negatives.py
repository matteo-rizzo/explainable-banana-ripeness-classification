from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeClassifier

from src.cv.classifiers.deep_learning.functional.yaml_manager import load_yaml
from src.nlp.dataset import train_val_test, wrong_predictions
from src.nlp.simple_model.pipeline import naive_classifier, predict_scores

classifier_type = RidgeClassifier

if __name__ == "__main__":
    out = Path("dumps") / "nlp_models" / "error_reports" / "vanilla"

    train_config: dict = load_yaml("src/nlp/params/experiment.yml")
    clf_params = train_config[classifier_type.__name__]
    synthetic_add: bool = True  # train_config["add_synthetic"]
    task: str = train_config["task"]

    # Create dataset
    data = train_val_test(target="M", add_synthetic_train=synthetic_add)

    # Train model
    _, pipe_m = naive_classifier(classifier_type(**clf_params), data, return_pipe=True, predict=False)

    x_data = data["test"]["x"] + data["test_synt"]["x"]
    y_data = data["test"]["y"] + data["test_synt"]["y"]

    # Tokenize dataset, then extract non-zero entries from vectorizer to get the effective features (words) that are considered
    x_tokenized = pipe_m["vectorizer"].transform(x_data)
    x_cleaned = pipe_m["vectorizer"].inverse_transform(x_tokenized)
    x_cleaned = [" ".join(ts) for ts in x_cleaned]

    # Predict scores with the model on test data
    m_scores = predict_scores(pipe_m, x_data)

    # Find out which are wrong predictions
    error_df: pd.DataFrame = wrong_predictions(y_pred=m_scores, y_true=np.asarray(y_data, dtype=int), threshold=.0)

    # Concatenate to wrong samples its input text
    input_df = pd.DataFrame({"original_text": x_data, "features": x_cleaned})
    input_df = input_df.iloc[error_df["indices"], :].reset_index(drop=True)
    error_df = pd.concat([error_df, input_df], axis=1)  # concat columns (same number of rows)

    # Separate errors in FP and FN and write reports to file
    error_df_fp = error_df[error_df["type"] == "fp"]
    error_df_fn = error_df[error_df["type"] == "fn"]

    out.mkdir(parents=True, exist_ok=True)
    error_df_fp.to_csv(out / "errors_fp.csv", index=False)
    error_df_fn.to_csv(out / "errors_fn.csv", index=False)
