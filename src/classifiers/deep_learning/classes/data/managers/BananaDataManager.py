import os
from pathlib import Path
from typing import Dict, Union

import numpy as np
from sklearn.model_selection import ShuffleSplit

from src.classifiers.deep_learning.classes.data.managers.BaseDataManager import BaseDataManager


class BananaDataManager(BaseDataManager):

    def __init__(self, data_params: Dict):
        super().__init__(data_params)
        self._data = self._read_data()

    def get_k(self) -> int:
        return super(BananaDataManager, self).get_k()

    def _read_data(self) -> Dict:
        data = {}
        for label in os.listdir(self._path_to_images):
            # Mac users rejoice
            if label == ".DS_Store":
                continue
            path_to_class_dir = os.path.join(self._path_to_images, label)
            items = os.listdir(path_to_class_dir)
            if ".DS_Store" in str(path_to_class_dir):
                continue
            data[label] = {}
            data[label]['x_paths'] = np.array([os.path.join(path_to_class_dir, item) for item in items])
            data[label]['y'] = np.array([int(label)] * len(data[label]['x_paths']))
        return data

    def generate_split(self):

        print(" Generating new splits...")

        ss = ShuffleSplit(n_splits=self._k, test_size=1 / self._k, random_state=0)

        for i in range(self._k):

            x_train_paths, x_val_paths, x_test_paths = [], [], []
            y_train, y_valid, y_test = [], [], []

            for label in self._data.keys():
                x_paths, y = self._data[label]['x_paths'], self._data[label]['y']

                train_val, test = list(ss.split(x_paths, y))[i]
                train, val = list(ss.split(x_paths[train_val], y[train_val]))[i]

                x_train_paths.extend(list(x_paths[train_val][train])), y_train.extend(list(y[train_val][train]))
                x_val_paths.extend(list(x_paths[train_val][val])), y_valid.extend(list(y[train_val][val]))
                x_test_paths.extend(list(x_paths[test])), y_test.extend(list(y[test]))

            self._split_data.append({
                'train': (x_train_paths, y_train),
                'val': (x_val_paths, y_valid),
                'test': (x_test_paths, y_test)
            })

    def reload_split(self, path_to_metadata: str, seed: int):
        super(BananaDataManager, self).reload_split(path_to_metadata, seed)

    def print_split_info(self, verbose: bool = False):
        return super(BananaDataManager, self).print_split_info(verbose)

    def load_split(self, fold: int) -> Dict:
        return super(BananaDataManager, self).load_split(fold)

    def save_split_to_file(self, path_to_results: Union[str, Path], seed: int):
        super(BananaDataManager, self).save_split_to_file(path_to_results, seed)
