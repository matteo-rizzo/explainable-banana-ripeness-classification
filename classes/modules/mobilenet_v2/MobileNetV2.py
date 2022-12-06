from typing import Dict

import torch
import torch.nn as nn
from torchvision import transforms


class MobileNetV2(nn.Module):

    def __init__(self, network_params: Dict):
        super().__init__()
        self.__mobilenet_v2 = torch.hub.load(repo_or_dir=network_params["repo_or_dir"],
                                             model=network_params["pretrained_model"],
                                             weights=network_params["weights"])
        self.__classifier = nn.Linear(network_params["in_features"],
                                      network_params["output_size"])
        self.__normalize = transforms.Normalize(network_params["normalization"]["mean"],
                                                network_params["normalization"]["std"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__normalize(x)
        x = self.__mobilenet_v2(x)
        x = self.__classifier(x)
        return x
