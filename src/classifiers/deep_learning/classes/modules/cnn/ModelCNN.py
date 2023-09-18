from typing import Dict

import torch

from src.classifiers.deep_learning.classes.core.Model import Model
from src.classifiers.deep_learning.classes.modules.cnn.CNN import CNN


class ModelCNN(Model):

    def __init__(self, network_params: Dict):
        super().__init__(device=network_params["device"])
        self._network = CNN(network_params).to(self._device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._network(x)
