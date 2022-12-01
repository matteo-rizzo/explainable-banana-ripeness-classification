from pathlib import Path

import torch

from classes.data.DataManager import DataManager
from classes.explainability.ModelSHAP import ModelSHAP
from classes.factories.ModelFactory import ModelFactory
from classes.utils.Params import Params
from utilities.yaml_manager import load_yaml

experiment_name = "treviso-market-224_224_pre_trained_vit_Thu_Dec__1_15-27-23_2022"
seed_n = 1
model_pth = "pre_trained_vit_fold_0.pth"


def explain_main():
    path_dump = Path("results") / experiment_name
    model_path = path_dump / f"seed_{seed_n}" / "models" / model_pth
    experiment_params = load_yaml(path_dump / "experiment.yml")
    network_params = load_yaml(path_dump / "network_params.yml")
    data_params = {
        "dataset": load_yaml(path_dump / "data.yml"),
        "cv": experiment_params["cv"],
        "batch_size": network_params["batch_size"]
    }

    # Setup devices and seeds for training
    device = torch.device("cpu")  # get_device(experiment_params["device"])

    network_type = network_params["architecture"]
    dataset_name = data_params["dataset"]["name"]

    data_manager = DataManager(data_params)
    data_manager.reload_split(str(path_dump / "cv_splits"), seed=1)
    data = data_manager.load_split(fold=1)

    network_params = Params.load_network_params(network_type)
    network_params["device"] = device

    model = ModelFactory().get(network_type, network_params)
    model.load(model_path)

    shap_model = ModelSHAP(model._network, device)
    shap_model.explain(data["test"])


if __name__ == "__main__":
    explain_main()
