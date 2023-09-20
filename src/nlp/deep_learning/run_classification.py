from src.cv.classifiers.deep_learning.functional.yaml_manager import load_yaml
from src.nlp.dataset import train_val_test, compute_metrics
from src.nlp.deep_learning.pipeline import create_hf_pipeline

if __name__ == "__main__":
    config: dict = load_yaml("src/nlp/params/deep_learning.yml")
    bs: int = config["batch_size"]
    target_label: str = config["target_label"]

    pipe = create_hf_pipeline(config["model_name"], device=0, batch_size=bs)

    dataset1 = train_val_test(target="M")

    results = pipe(dataset1["test"]["x"])
    results = [1 if e["label"] == target_label else 0 for e in results]
    metrics = compute_metrics(y_pred=results, y_true=dataset1["test"]["y"])
    print(metrics)
    m_f1 = metrics["f1"]

    dataset2 = train_val_test(target="A")

    results = pipe(dataset2["test"]["x"])
    results = [1 if e["label"] == "hateful" else 0 for e in results]
    metrics = compute_metrics(y_pred=results, y_true=dataset2["test"]["y"])
    print(metrics)
    a_f1 = metrics["f1"]

    a_score = (m_f1 + a_f1) / 2
    print(f"\n*** Task A score: {a_score:.5f} ***")
