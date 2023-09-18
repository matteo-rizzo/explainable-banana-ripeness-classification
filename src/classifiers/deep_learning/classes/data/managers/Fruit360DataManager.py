import os
from pathlib import Path
from typing import Dict, Union

import numpy as np
from sklearn.model_selection import ShuffleSplit

from src.classifiers.deep_learning.classes.data.managers.BaseDataManager import BaseDataManager


class Fruit360DataManager(BaseDataManager):

    def __init__(self, data_params: Dict):
        super().__init__(data_params)
        train_folder: str = data_params["dataset"]["paths"]["train_split"]
        test_folder: str = data_params["dataset"]["paths"]["test_split"]
        self.label_map: Dict[str, Dict[int, str]] = {train_folder: {}, test_folder: {}}
        self._train_data = self._read_data(train_folder)
        self._test_data = self._read_data(test_folder)
        # Verify the label maps in the disjoint sets are the same
        assert self.label_map["Training"] == self.label_map["Test"]

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
            self.label_map[folder][label_index] = label
            data[label_index]['y'] = np.array([label_index] * len(data[label_index]['x_paths']))
            label_index += 1
        # Test
        return data

    def generate_split(self):

        print(" Generating new splits...")

        ss = ShuffleSplit(n_splits=self._k, test_size=1 / self._k, random_state=0)

        for i in range(self._k):

            x_train_paths, x_val_paths, x_test_paths = [], [], []
            y_train, y_valid, y_test = [], [], []

            for label in self._train_data.keys():
                x_paths, y = self._train_data[label]['x_paths'], self._train_data[label]['y']

                train, val = list(ss.split(x_paths, y))[i]
                x_train_paths.extend(list(x_paths[train])), y_train.extend(list(y[train]))
                x_val_paths.extend(list(x_paths[val])), y_valid.extend(list(y[val]))

                # Always the same, as the test set is predefined
                x_test_paths.extend(list(self._test_data[label]['x_paths']))
                y_test.extend(list(self._test_data[label]['y']))

            self._split_data.append({
                'train': (x_train_paths, y_train),
                'val': (x_val_paths, y_valid),
                'test': (x_test_paths, y_test)
            })

    def reload_split(self, path_to_metadata: str, seed: int):
        super(Fruit360DataManager, self).reload_split(path_to_metadata, seed)

    def print_split_info(self, verbose: bool = False):
        return super(Fruit360DataManager, self).print_split_info(verbose)

    def load_split(self, fold: int) -> Dict:
        return super(Fruit360DataManager, self).load_split(fold)

    def save_split_to_file(self, path_to_results: Union[str, Path], seed: int):
        super(Fruit360DataManager, self).save_split_to_file(path_to_results, seed)
