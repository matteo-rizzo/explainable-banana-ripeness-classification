from pprint import pprint

import pandas as pd
import torch.cuda

from src.cv.classifiers.deep_learning.functional.yaml_manager import load_yaml
from src.nlp.dataset import train_val_test, compute_metrics, task_b_eval
from src.nlp.deep_learning.pipeline import create_hf_pipeline

if __name__ == "__main__":
    config: dict = load_yaml("src/nlp/params/deep_learning.yml")
    bs: int = config["training"]["test_batch_size"]
    target_label: str = config["testing"]["target_label"]
    use_gpu: bool = config["use_gpu"]
    task: str = config["task"]
    add_synthetic: bool = config["add_synthetic"]

    print("*** Predicting misogyny ")
    pipe_m = create_hf_pipeline(config["testing"]["task_m_model_name"], device=0 if use_gpu else "cpu", batch_size=bs, top_k=1)
    dataset_m = train_val_test(target="M", add_synthetic_train=add_synthetic or task == "B")
    results = pipe_m(dataset_m["test"]["x"])
    results = [1 if e[0]["label"] == target_label else 0 for e in results]
    metrics = compute_metrics(y_pred=results, y_true=dataset_m["test"]["y"])
    pprint(metrics)
    m_f1 = metrics["f1"]

    match task:
        case "B":
            print("*** Task B ")
            test_synt = dataset_m["test_synt"]
            m_synt_pred = pipe_m(test_synt["x"])
            m_synt_pred = [1 if e[0]["label"] == target_label else 0 for e in m_synt_pred]

            df_pred = pd.Series(results, index=pd.Index(dataset_m["test"]["ids"], dtype=str))
            df_pred_synt = pd.Series(m_synt_pred, index=pd.Index(test_synt["ids"], dtype=str))

            # Preparing data for evaluation
            df_pred = df_pred.to_frame().reset_index().rename(columns={"index": "id", 0: "misogynous"})
            df_pred_synt = df_pred_synt.to_frame().reset_index().rename(columns={"index": "id", 0: "misogynous"})
            # Read gold test set
            task_b_eval(dataset_m, df_pred, df_pred_synt)

        case "A":
            print("*** Task A")
            print("*** Predicting aggressiveness ")

            # Free memory
            del pipe_m
            del results
            torch.cuda.empty_cache()

            pipe_a = create_hf_pipeline(config["testing"]["task_a_model_name"], device=0 if use_gpu else "cpu", batch_size=bs, top_k=1)
            dataset_a = train_val_test(target="A")
            results = pipe_a(dataset_a["test"]["x"])
            results = [1 if e[0]["label"] == target_label else 0 for e in results]
            metrics = compute_metrics(y_pred=results, y_true=dataset_a["test"]["y"])
            print(metrics)
            a_f1 = metrics["f1"]

            # TASK A evaluation (misogynous + aggressiveness)
            a_score = (m_f1 + a_f1) / 2
            print(f"\n*** Task A score: {a_score:.5f} ***")
