from datasets import load_dataset, Dataset, NamedSplit, Features, Value

from src.nlp.dataset import train_val_test


def create_hf_dataset() -> tuple[Dataset, Dataset]:
    dataset = train_val_test(target="M")

    data_hf = {k: {"text": v["x"], "label": v["y"]} for k, v in dataset.items()}

    feat = Features({
        "text": Value("string"),
        "label": Value("int8")}
    )

    ds_train = Dataset.from_dict(data_hf["train"], split=NamedSplit("train"), features=feat)
    ds_test = Dataset.from_dict(data_hf["test"], split=NamedSplit("test"), features=feat)

    return ds_train, ds_test
