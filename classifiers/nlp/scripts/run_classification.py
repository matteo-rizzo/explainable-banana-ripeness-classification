# Read data
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.linear_model import RidgeClassifier

from classifiers.nlp.scripts.pipeline import compute_metrics, train_val_test, naive_classifier
from src.classifiers.deep_learning.functional.yaml_manager import load_yaml

classifier_type = RidgeClassifier

if __name__ == "__main__":
    train_config: dict = load_yaml("params/experiment.yml")
    clf_params = train_config[classifier_type.__name__]

    print("*** Misogyny task")
    data = train_val_test(target="M")
    m_pred = naive_classifier(classifier_type(**clf_params), data)
    m_f1 = compute_metrics(m_pred, data["test"]["y"], classifier_type.__name__)["f1"]

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

    a_pred = naive_classifier(classifier_type(**clf_params), data)
    a_true = data["test"]["y"]
    a_ids = data["test"]["ids"]
    # a_pred = [0] * len(m_pred)

    # a_pred = np.concatenate([a_pred, np.array([0] * len(non_misogyny_ids))])
    # a_ids = data_aggressiveness["test"]["ids"] + list(non_misogyny_ids)
    # a_true = data_aggressiveness["test"]["y"] + ([0] * len(non_misogyny_ids))

    a_f1 = compute_metrics(a_pred, a_true, classifier_type.__name__)["f1"]

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
