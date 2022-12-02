import os
import pprint
from pathlib import Path
from typing import Dict, Union

import pandas as pd
from termcolor import colored
from torch.utils.data import DataLoader

from classes.data.Dataset import Dataset
from classes.data.loaders.ImageLoader import ImageLoader


class DataManager:
    def __init__(self, data_params: Dict):
        super().__init__()
        self._path_to_images = Path(data_params["dataset"]["paths"]["images"])
        self._k = data_params["cv"]["k"]
        self._batch_size = data_params["batch_size"]
        self._loader = ImageLoader().load
        self._split_data = []

    def get_k(self) -> int:
        return self._k

    def _read_data(self, **kwargs):
        # Abstract method
        raise NotImplementedError

    def generate_split(self):
        # Abstract method
        raise NotImplementedError

    def reload_split(self, path_to_metadata: str, seed: int):
        """ Reloads the data split from saved metadata in CSV format """
        saved_data = pd.read_csv(os.path.join(path_to_metadata, f"split_{seed}.csv"),
                                 converters={"train": eval, "val": eval, "test": eval})
        self._split_data = saved_data.to_dict("records")

    def print_split_info(self, verbose: bool = False):
        """ Shows how the data has been split_data in each fold """

        split_info = []
        for fold_paths in self._split_data:
            split_info.append({
                'train': list(set([item.split("_")[-1] for item in fold_paths['train'][0]])),
                'val': list(set([item.split("_")[-1] for item in fold_paths['val'][0]])),
                'test': list(set([item.split("_")[-1] for item in fold_paths['test'][0]]))
            })

        if verbose:
            print("\n..............................................\n")
            print("Split info overview:\n")
            pp = pprint.PrettyPrinter(compact=True)
            for fold in range(self._k):
                print(colored(f'fold {fold}: ', 'blue'))
                pp.pprint(split_info[fold])
                print('\n')
            print("\n..............................................\n")

    def load_split(self, fold: int) -> Dict:
        """ Loads the data based on the fold paths """

        fold_paths = self._split_data[fold]

        x_train_paths, y_train = fold_paths['train']
        x_val_paths, y_valid = fold_paths['val']
        x_test_paths, y_test = fold_paths['test']

        print("\n..............................................")
        print("Split size overview:")
        for set_type, y in {"train": y_train, "val": y_valid, "test": y_test}.items():
            print(f"\t * {set_type.upper()}: {len(y)}")
        print("..............................................")

        return {
            'train': DataLoader(Dataset(x_train_paths, y_train, self._loader),
                                self._batch_size, shuffle=True, drop_last=True),
            'val': DataLoader(Dataset(x_val_paths, y_valid, self._loader),
                              self._batch_size, drop_last=True),
            'test': DataLoader(Dataset(x_test_paths, y_test, self._loader),
                               self._batch_size, drop_last=True)
        }

    def save_split_to_file(self, path_to_results: Union[str, Path], seed: int):
        path_to_metadata = os.path.join(path_to_results, "cv_splits")
        path_to_file = os.path.join(path_to_metadata, f"split_{seed}.csv")
        os.makedirs(path_to_metadata, exist_ok=True)
        pd.DataFrame(self._split_data).to_csv(path_to_file, index=False)
        print(f" CV split metadata written at {path_to_file}")
