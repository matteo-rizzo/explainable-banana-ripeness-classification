import os

import torch
from pathlib import Path
from classes.utils.Params import Params


class Loader:

    def __init__(self, modality: str):
        self._modality = modality
        self._modality_params = Params.load_modality_params(self._modality)

        experiment_params = Params.load_experiment_params()
        dataset_params = Params.load_dataset_params(experiment_params["dataset_name"])
        self._path_to_modalities = dataset_params["paths"]
        self._network_type = experiment_params["train"]["network_type"]

        path_to_modality = self._path_to_modalities[self._modality]
        self._path_to_data = os.path.join(path_to_modality, self._modality_params["path_to_data"])
        self._file_format = self._modality_params["file_format"]

    def _get_path_to_item(self, path_to_input: str) -> str:
        """
        Creates the path to the data item for the specified modality
        :param path_to_input: the path to the data item related to the main modality
        :return: the path to the data item
        """
        # TODO: I am confused at the purpose of this function.
        # TODO: It returns the same thing it gets, but with extra steps
        # path_to_input = Path(path_to_input)
        # file_name = path_to_input.name
        # label = path_to_input.parent.name
        # # split_path = path_to_input.split(os.sep)
        # # file_name = str(split_path[-1]).split(".")[0] + "." + self._file_format
        # # label = str(split_path[-2])
        # return os.path.join(self._path_to_data, label, file_name)
        return path_to_input

    def load(self, path_to_input: str) -> torch.Tensor:
        """
        Loads a data item from the dataset
        :param path_to_input: the path to the data item to be loaded (referred to the main modality)
        :return: the loaded data item
        """
        pass
