import os
from pathlib import Path
from typing import Dict, Union

import numpy as np

from classes.data.managers.DataManager import DataManager


class Fruit360DataManager(DataManager):

    def __init__(self, data_params: Dict):
        super().__init__(data_params)
        train_folder: str = data_params["dataset"]["paths"]["train_split"]
        test_folder: str = data_params["dataset"]["paths"]["test_split"]
        self.label_map: Dict[int, str] = {}
        self._train_data = self._read_data(train_folder)
        self._test_data = self._read_data(test_folder)

    def get_k(self) -> int:
        return super(Fruit360DataManager, self).get_k()

    def _read_data(self, folder) -> Dict:
        data = {}
        # Train
        label_index: int = 0
        for label in os.listdir(self._path_to_images / folder):
            # Mac users rejoice
            if label == ".DS_Store":
                continue
            path_to_class_dir = self._path_to_images / folder / label
            items = os.listdir(path_to_class_dir)
            if ".DS_Store" in str(path_to_class_dir):
                continue
            data[label_index] = {}
            data[label_index]['x_paths'] = np.array([os.path.join(path_to_class_dir, item) for item in items])
            self.label_map[label_index] = label
            data[label_index]['y'] = np.array([label_index] * len(data[label_index]['x_paths']))
            label_index += 1
        # Test
        return data

    def generate_split(self):
        super(Fruit360DataManager, self).generate_split()

    def reload_split(self, path_to_metadata: str, seed: int):
        super(Fruit360DataManager, self).reload_split(path_to_metadata, seed)

    def print_split_info(self, verbose: bool = False):
        return super(Fruit360DataManager, self).print_split_info(verbose)

    def load_split(self, fold: int) -> Dict:
        return super(Fruit360DataManager, self).load_split(fold)

    def save_split_to_file(self, path_to_results: Union[str, Path], seed: int):
        super(Fruit360DataManager, self).save_split_to_file(path_to_results, seed)
