# Read data
from pathlib import Path

import pandas as pd
from sklearn.linear_model import RidgeClassifier

from src.cv.classifiers.deep_learning.functional.yaml_manager import load_yaml
from src.nlp.custom_vocab.custom_vocab import custom_vocab_classifier
from src.nlp.dataset import train_val_test, compute_metrics, task_b_eval
from src.nlp.simple_model.pipeline import naive_classifier

classifier_type = RidgeClassifier

TEAM_NAME = "myTeam"


def create_ami_submission(predictions: pd.DataFrame | pd.Series, task_type: str,
                          data_type: str, run_type: str, run_id: str, team_name: str) -> str:
    f = f"results/{team_name}.{task_type.upper()}.{data_type}.{run_type}.{run_id}"
    Path(f).parent.mkdir(exist_ok=True, parents=True)

    predictions.to_csv(f, header=False, sep="\t")
    return f


if __name__ == "__main__":
    train_config: dict = load_yaml("src/nlp/params/experiment.yml")
    clf_params = train_config[classifier_type.__name__]
    synthetic_add: bool = train_config["add_synthetic"]
    task: str = train_config["task"]

    print("*** Predicting misogyny ")
    data = train_val_test(target="M", add_synthetic_train=synthetic_add)
    m_pred, clf_m, vec_m = custom_vocab_classifier(classifier_type(**clf_params), data,
                                                   max_features=train_config["max_features"])
    m_f1 = compute_metrics(m_pred, data["test"]["y"], classifier_type.__name__)["f1"]

    match task:
        case "B":
            print("*** Task B ")
            if not synthetic_add:
                test_synt: dict = train_val_test(target="M", add_synthetic_train=True)["test_synt"]
            else:
                test_synt = data["test_synt"]
            m_synt_pred = clf_m.predict(vec_m.transform(test_synt["x"]).toarray())

            df_pred = pd.Series(m_pred, index=pd.Index(data["test"]["ids"], dtype=str))
            df_pred_synt = pd.Series(m_synt_pred, index=pd.Index(test_synt["ids"], dtype=str))

            # Preparing data for evaluation
            df_pred = df_pred.to_frame().reset_index().rename(columns={"index": "id", 0: "misogynous"})
            df_pred_synt = df_pred_synt.to_frame().reset_index().rename(columns={"index": "id", 0: "misogynous"})
            # Read gold test set
            task_b_eval(data, df_pred, df_pred_synt)
        case "A":

            print("*** Task A")
            print("*** Predicting aggressiveness ")
            data = train_val_test(target="A")

            a_pred = naive_classifier(classifier_type(**clf_params), data)
            a_true = data["test"]["y"]
            a_ids = data["test"]["ids"]

            a_f1 = compute_metrics(a_pred, a_true, classifier_type.__name__)["f1"]

            a_score = (m_f1 + a_f1) / 2
            print(f"\n*** Task A score: {a_score:.5f} ***")

        case _:
            raise ValueError(f"Unsupported task '{task}'. Only 'A' or 'B' are possible values.")

    # Best task A score: 0.707
