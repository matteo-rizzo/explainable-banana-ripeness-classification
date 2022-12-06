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

    # def _get_path_to_item(self, path_to_input: str) -> str:
    #     """
    #     Creates the path to the data item for the specified modality
    #     :param path_to_input: the path to the data item related to the main modality
    #     :return: the path to the data item
    #     """
    #     # TODO: I am confused at the purpose of this function.
    #     # TODO: It returns the same thing it gets, but with extra steps
    #     # path_to_input = Path(path_to_input)
    #     # file_name = path_to_input.name
    #     # label = path_to_input.parent.name
    #     # # split_path = path_to_input.split(os.sep)
    #     # # file_name = str(split_path[-1]).split(".")[0] + "." + self._file_format
    #     # # label = str(split_path[-2])
    #     # return os.path.join(self._path_to_data, label, file_name)
    #     return path_to_input

    def load(self, path_to_input: str) -> torch.Tensor:
        """
        Loads a data item from the dataset
        :param path_to_input: the path to the data item to be loaded (referred to the main modality)
        :return: the loaded data item
        """
        raise NotImplementedError
