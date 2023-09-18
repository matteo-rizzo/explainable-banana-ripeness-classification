import os
from pathlib import Path
from typing import Tuple, Dict, Union

import pandas as pd
import yaml


class Params:
    base_param_path: Path = Path("params")

    @staticmethod
    def load() -> Tuple:
        experiment_params = Params.load_experiment_params()
        train_params = experiment_params["train"]
        data_params = {
            "dataset": Params.load_dataset_params(experiment_params["dataset_name"]),
            "cv": experiment_params["cv"],
            "batch_size": train_params["batch_size"]
        }
        train_params["num_classes"] = data_params["dataset"]["num_classes"]
        num_seeds = experiment_params["num_seeds"]
        device_type = experiment_params["device"]
        return train_params, data_params, num_seeds, device_type

    @staticmethod
    def load_experiment_params() -> Dict:
        """
        Loads the parameters stored in the params/experiment.yaml file
        :return: the loaded experiment parameters in a Dict
        """
        experiments_path: Path = Params.base_param_path / "experiment.yml"
        with open(experiments_path, "r") as f:
            return yaml.load(f, Loader=yaml.SafeLoader)

    @staticmethod
    def load_network_params(network_type: str) -> Dict:
        """
        Loads and preprocesses the parameters stored in the params/modules/network_type.yaml file
        :param network_type: the type of network to be loaded
        :return: the loaded network parameters in a Dict
        """
        path_to_params: Path = Params.base_param_path / "networks" / f"{network_type}.yml"
        if not path_to_params.is_file():
            raise ValueError(f"Params file '{str(path_to_params)}' for network '{network_type}' not found!"
                             f"\n Available params files are: {os.listdir(Params.base_param_path / 'networks')}")
        network_params = yaml.load(open(path_to_params, "r"), Loader=yaml.SafeLoader)
        network_params["architecture"] = network_type
        network_params["batch_size"] = Params.load_experiment_params()["train"]["batch_size"]
        return network_params

    @staticmethod
    def load_dataset_params(dataset_name: str) -> Dict:
        """
        Loads the parameters stored in the params/modules/dataset_name.yml file merging the paths
        :param dataset_name: the type of data to be loaded
        :return: the loaded data parameters in a Dict
        """
        path_to_params: Path = Params.base_param_path / "dataset" / f"{dataset_name}.yml"
        if not path_to_params.is_file():
            raise ValueError(
                f"Params file '{path_to_params}' for dataset '{dataset_name}' not found! "
                f"\n Available params files are: {os.listdir(Params.base_param_path / 'dataset')}")
        with open(path_to_params, "r") as f:
            params = yaml.load(f, Loader=yaml.SafeLoader)
        params["name"] = dataset_name

        dataset_dir = params["paths"]["dataset_dir"]
        params["paths"]["dataset_metadata"] = os.path.join(dataset_dir, params["paths"]["dataset_metadata"])
        params["paths"]["cv_metadata"] = os.path.join(dataset_dir, params["paths"]["cv_metadata"])

        return params

    @staticmethod
    def save(data: Dict, path_to_destination):
        """
        Saves the given data into a YAML file at the given destination
        :param data: the data to be saved
        :param path_to_destination: the destination of the file with the saved metrics
        """
        with open(path_to_destination, 'w') as f:
            yaml.dump(data, f, Dumper=yaml.SafeDumper, default_flow_style=False)

    @staticmethod
    def save_experiment_params(path_to_results: Union[str, Path], network_type: str, dataset_name: str):
        """
        Saves the configuration for the current experiment to file at the given path
        :param path_to_results: the path where to save the configuration of the experiment at
        :param network_type: the type of network used for the experiment
        :param dataset_name: the type of data used for the experiment
        """
        Params.save(Params.load_experiment_params(), os.path.join(path_to_results, "experiment.yml"))
        Params.save(Params.load_dataset_params(dataset_name), os.path.join(path_to_results, "data.yml"))
        Params.save(Params.load_network_params(network_type), os.path.join(path_to_results, "network_params.yml"))

    @staticmethod
    def save_experiment_preds(fold_evaluation: Dict, path_to_preds: str, fold_number: int):
        """
        Saves the experiments preds in CSV format at the given path
        :param fold_evaluation: the evaluation data for the given fold, including ground truth and preds
        :param path_to_preds: the path where to store the preds at
        :param fold_number: the number of the fold for generating the name of the file
        """
        for set_type in ["train", "val", "test"]:
            path_to_csv = os.path.join(path_to_preds, "fold_" + str(fold_number) + "_" + set_type + "_preds.csv")
            pd.DataFrame(fold_evaluation["preds"][set_type]).to_csv(path_to_csv, index=False)
