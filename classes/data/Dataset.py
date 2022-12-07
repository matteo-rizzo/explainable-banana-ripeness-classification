from typing import List

import torch
from torch.utils.data import DataLoader


class Dataset(torch.utils.data.Dataset):

    def __init__(self, split: str, data_paths: List, labels: List, loader: callable, file_names: List = None):
        """
        @param data_paths: list of paths to sequences
        @param labels: list of corresponding labels
        @param loader: transform to be applied on a data item
        """
        self.__split: str = split
        self.__data_paths = data_paths
        self.__labels = labels
        self.__loader = loader
        self.__file_names = file_names

    def __len__(self):
        return len(self.__labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x, y = self.__data_paths[idx], self.__labels[idx]

        if self.__file_names is not None:
            fn = self.__file_names[idx]
            return self.__loader(x, self.__split), y, fn

        return self.__loader(x, self.__split), y
