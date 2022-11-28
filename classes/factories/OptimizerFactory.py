from typing import List

import torch
from torch import optim


class OptimizerFactory:

    def __init__(self, network_parameters: List, learning_rate: float):
        self.optimizers_map = {
            "SGD": optim.SGD(network_parameters, lr=learning_rate, momentum=0.9, nesterov=True),
            "Adam": optim.Adam(network_parameters, lr=learning_rate),
            "AdamW": optim.AdamW(network_parameters, lr=learning_rate),
        }

    def get(self, optimizer_type: str) -> torch.optim:
        if optimizer_type not in self.optimizers_map.keys():
            raise ValueError(
                f"Optimizer for {optimizer_type} is not implemented! "
                f"\n Implemented optimizers are: {list(self.optimizers_map.keys())}")
        return self.optimizers_map[optimizer_type]
