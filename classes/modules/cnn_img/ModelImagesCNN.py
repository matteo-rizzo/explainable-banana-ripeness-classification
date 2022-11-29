from typing import Dict

import torch
from torchvision.transforms import transforms

from classes.core.Model import Model
from classes.modules.cnn_img.ImagesCNN import ImagesCNN


class ModelImagesCNN(Model):

    def __init__(self, network_params: Dict):
        super().__init__(device=network_params["device"])
        self._network = ImagesCNN(network_params).to(self._device)
        self.__normalization = network_params["normalization"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # --- Normalization ---
        # This variant normalizes here to use faster gpu matrix operations
        mean, std = self.__normalization.values()
        normalization = transforms.Normalize(mean=mean, std=std)
        x = normalization(x)
        # ---------------------
        return self.model(x)

