from typing import Dict

from classes.data.managers import BananaDataManager, Fruit360DataManager, DataManager


class DataManagerFactory:
    data_managers_map = {
        "bananas": BananaDataManager,
        "fruit-360": Fruit360DataManager
    }

    def get(self, data_params: Dict) -> DataManager:
        manager = data_params['dataset']['manager']
        if manager not in self.data_managers_map.keys():
            raise ValueError(
                f"Data manager {manager} is not implemented! "
                f"\n Implemented Data managers are: {list(self.data_managers_map.keys())}")
        return self.data_managers_map[manager](data_params)
