from pathlib import Path
from typing import Dict, Union

from classes.data.managers.DataManager import DataManager


class BananaDataManager(DataManager):

    def __init__(self, data_params: Dict):
        super().__init__(data_params)

    def get_k(self) -> int:
        return super(BananaDataManager, self).get_k()

    def _read_data(self) -> Dict:
        return super(BananaDataManager, self)._read_data()

    def generate_split(self):
        super(BananaDataManager, self).generate_split()

    def reload_split(self, path_to_metadata: str, seed: int):
        super(BananaDataManager, self).reload_split(path_to_metadata, seed)

    def print_split_info(self, verbose: bool = False):
        return super(BananaDataManager, self).print_split_info(verbose)

    def load_split(self, fold: int) -> Dict:
        return super(BananaDataManager, self).load_split(fold)

    def save_split_to_file(self, path_to_results: Union[str, Path], seed: int):
        super(BananaDataManager, self).save_split_to_file(path_to_results, seed)
