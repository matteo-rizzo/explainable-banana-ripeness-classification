import torch


class Loader:

    def load(self, path_to_input: str, split: str) -> torch.Tensor:
        """
        Loads a data item from the dataset
        :param path_to_input: the path to the data item to be loaded (referred to the main modality)
        :param split: For augmentation purposes [train, test, val, ignore]
        :return: the loaded data item
        """
        raise NotImplementedError
