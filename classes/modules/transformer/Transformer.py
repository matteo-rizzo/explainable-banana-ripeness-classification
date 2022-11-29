from typing import Dict

import torch
import torch.nn as nn
from torchvision.transforms import transforms


class Transformer(nn.Module):

    def __init__(self, network_params: Dict):
        super().__init__()
        self.__normalization = network_params["normalization"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Normalization ---
        # This variant normalizes here to use faster gpu matrix operations
        mean, std = self.__normalization.values()
        normalization = transforms.Normalize(mean=mean, std=std)
        x = normalization(x)
        # ---------------------
        return self.model(x)
