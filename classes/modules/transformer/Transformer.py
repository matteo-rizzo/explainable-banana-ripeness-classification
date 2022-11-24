from typing import Dict

import torch
import torch.nn as nn


class Transformer(nn.Module):

    def __init__(self, network_params: Dict):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.__model(x)
