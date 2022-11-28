import os
import pprint
from typing import Dict, Union
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from termcolor import colored
from torch.utils.data import DataLoader

from classes.data.Dataset import Dataset
from classes.data.loaders.ImageLoader import ImageLoader


class DataManager:

    def __init__(self, data_params: Dict, network_type: str):
        self.__path_to_images = data_params["dataset"]["paths"]["images"]
        self.__k = data_params["cv"]["k"]
        self.__batch_size = data_params["batch_size"]
        self.__loader = ImageLoader().load
        self.__split_data = []
        self.__data = self.__read_data()

    def get_k(self) -> int:
        return self.__k

    def __read_data(self) -> Dict:
        data = {}
        for label in os.listdir(self.__path_to_images):
            if label == ".DS_Store":
                continue
            path_to_class_dir = os.path.join(self.__path_to_images, label)
            items = os.listdir(path_to_class_dir)
            if ".DS_Store" in path_to_class_dir:
                continue
            data[label] = {}
            data[label]['x_paths'] = np.array([os.path.join(path_to_class_dir, item) for item in items])
            data[label]['y'] = np.array([int(label)] * len(data[label]['x_paths']))
        return data

    def generate_split(self):

        print(" Generating new splits...")

        ss = ShuffleSplit(n_splits=self.__k, test_size=1 / self.__k, random_state=0)

        for i in range(self.__k):

            x_train_paths, x_val_paths, x_test_paths = [], [], []
            y_train, y_valid, y_test = [], [], []

            for label in self.__data.keys():
                x_paths, y = self.__data[label]['x_paths'], self.__data[label]['y']

                train_val, test = list(ss.split(x_paths, y))[i]
                train, val = list(ss.split(x_paths[train_val], y[train_val]))[i]

                x_train_paths.extend(list(x_paths[train_val][train])), y_train.extend(list(y[train_val][train]))
                x_val_paths.extend(list(x_paths[train_val][val])), y_valid.extend(list(y[train_val][val]))
                x_test_paths.extend(list(x_paths[test])), y_test.extend(list(y[test]))

            self.__split_data.append({
                'train': (x_train_paths, y_train),
                'val': (x_val_paths, y_valid),
                'test': (x_test_paths, y_test)
            })

    def reload_split(self, path_to_metadata: str, seed: int):
        """ Reloads the data split from saved metadata in CSV format """
        saved_data = pd.read_csv(os.path.join(path_to_metadata, "split_{}.csv".format(seed)),
                                 converters={"train": eval, "val": eval, "test": eval})
        self.__split_data = saved_data.to_dict("records")

    def print_split_info(self):
        """ Shows how the data has been split_data in each fold """

        split_info = []
        for fold_paths in self.__split_data:
            split_info.append({
                'train': list(set([item.split("_")[-1] for item in fold_paths['train'][0]])),
                'val': list(set([item.split("_")[-1] for item in fold_paths['val'][0]])),
                'test': list(set([item.split("_")[-1] for item in fold_paths['test'][0]]))
            })
        # Note: this is TOO MUCH
        # print("\n..............................................\n")
        # print("Split info overview:\n")
        # pp = pprint.PrettyPrinter(compact=True)
        # for fold in range(self.__k):
        #     print(colored(f'fold {fold}: ', 'blue'))
        #     pp.pprint(split_info[fold])
        #     print('\n')
        # print("\n..............................................\n")

    def load_split(self, fold: int) -> Dict:
        """ Loads the data based on the fold paths """

        fold_paths = self.__split_data[fold]

        x_train_paths, y_train = fold_paths['train']
        x_val_paths, y_valid = fold_paths['val']
        x_test_paths, y_test = fold_paths['test']

        print("\n..............................................")
        print("Split size overview:")
        for set_type, y in {"train": y_train, "val": y_valid, "test": y_test}.items():
            num_pos = sum(y)
            num_neg = len(y) - num_pos
            print(f"\t * {set_type.upper()}: [ Pos: {num_pos} | Neg: {num_neg} ]")
        print("..............................................")

        return {
            'train': DataLoader(Dataset(x_train_paths, y_train, self.__loader), self.__batch_size, shuffle=True),
            'val': DataLoader(Dataset(x_val_paths, y_valid, self.__loader), self.__batch_size),
            'test': DataLoader(Dataset(x_test_paths, y_test, self.__loader), self.__batch_size)
        }

    def save_split_to_file(self, path_to_results: Union[str, Path], seed: int):
        path_to_metadata = os.path.join(path_to_results, "cv_splits")
        path_to_file = os.path.join(path_to_metadata, f"split_{seed}.csv")
        os.makedirs(path_to_metadata, exist_ok=True)
        pd.DataFrame(self.__split_data).to_csv(path_to_file, index=False)
        print(f" CV split metadata written at {path_to_file}")
