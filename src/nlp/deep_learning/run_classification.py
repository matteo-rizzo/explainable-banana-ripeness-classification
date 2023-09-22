import torch.cuda

from src.cv.classifiers.deep_learning.functional.yaml_manager import load_yaml
from src.nlp.dataset import train_val_test, compute_metrics
from src.nlp.deep_learning.pipeline import create_hf_pipeline

if __name__ == "__main__":
    config: dict = load_yaml("src/nlp/params/deep_learning.yml")
    bs: int = config["training"]["test_batch_size"]
    target_label: str = config["testing"]["target_label"]
    use_gpu: bool = config["use_gpu"]

    pipe_m = create_hf_pipeline(config["testing"]["task_m_model_name"], device=0 if use_gpu else "cpu", batch_size=bs, top_k=1)
    dataset_m = train_val_test(target="M")
    results = pipe_m(dataset_m["test"]["x"])
    results = [1 if e[0]["label"] == target_label else 0 for e in results]
    metrics = compute_metrics(y_pred=results, y_true=dataset_m["test"]["y"])
    print(metrics)
    m_f1 = metrics["f1"]

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
