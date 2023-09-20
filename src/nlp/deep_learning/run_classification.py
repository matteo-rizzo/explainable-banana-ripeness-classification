from src.cv.classifiers.deep_learning.functional.yaml_manager import load_yaml
from src.nlp.dataset import train_val_test, compute_metrics
from src.nlp.deep_learning.pipeline import create_hf_pipeline

if __name__ == "__main__":
    config: dict = load_yaml("src/nlp/params/deep_learning.yml")
    bs: int = config["training"]["test_batch_size"]
    target_label: str = config["testing"]["target_label"]

    pipe_m = create_hf_pipeline(config["testing"]["task_m_model_name"], device=0, batch_size=bs)
    dataset_m = train_val_test(target="M")
    results = pipe_m(dataset_m["test"]["x"])
    results = [1 if e["label"] == target_label else 0 for e in results]
    metrics = compute_metrics(y_pred=results, y_true=dataset_m["test"]["y"])
    print(metrics)
    m_f1 = metrics["f1"]

    pipe_a = create_hf_pipeline(config["testing"]["task_a_model_name"], device=0, batch_size=bs)
    dataset_a = train_val_test(target="A")
    results = pipe_a(dataset_a["test"]["x"])
    results = [1 if e["label"] == "hateful" else 0 for e in results]
    metrics = compute_metrics(y_pred=results, y_true=dataset_a["test"]["y"])
    print(metrics)
    a_f1 = metrics["f1"]

    a_score = (m_f1 + a_f1) / 2
    print(f"\n*** Task A score: {a_score:.5f} ***")

# {'f1': 0.7100990934008011, 'accuracy': 0.714, 'precision': 0.7261735660173161, 'recall': 0.714}
# {'f1': 0.4982942002809553, 'accuracy': 0.52, 'precision': 0.6005817099567099, 'recall': 0.6640556045895851}
