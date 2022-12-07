from typing import Dict

import torch

from classes.utils.Params import Params


class Loader:

    def __init__(self, img_details: Dict):
        experiment_params = Params.load_experiment_params()
        dataset_params = Params.load_dataset_params(experiment_params["dataset_name"])
        self._network_type = experiment_params["train"]["network_type"]

        self._path_to_data = dataset_params["paths"]["dataset_dir"]
        self._file_format = img_details["file_format"]

    def load(self, path_to_input: str, split: str) -> torch.Tensor:
        """
        Loads a data item from the dataset
        :param path_to_input: the path to the data item to be loaded (referred to the main modality)
        :param split: For augmentation purposes [train, test, val, ignore]
        :return: the loaded data item
        """
        raise NotImplementedError
