from datasets import Dataset, NamedSplit, Features, Value, ClassLabel

from src.nlp.dataset import train_val_test


def create_hf_dataset(target: str = "M") -> tuple[Dataset, Dataset]:
    dataset = train_val_test(target=target)

    data_hf = {k: {"text": v["x"], "label": v["y"]} for k, v in dataset.items()}

    feat = Features({
        "text": Value("string"),
        "label": ClassLabel(num_classes=2, names=["no", "yes"], names_file=None, id=None)}
    )

    ds_train = Dataset.from_dict(data_hf["train"], split=NamedSplit("train"), features=feat)
    ds_test = Dataset.from_dict(data_hf["test"], split=NamedSplit("test"), features=feat)

    return ds_train, ds_test
