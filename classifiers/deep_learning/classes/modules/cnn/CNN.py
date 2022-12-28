from typing import Dict

import torch
from torch import nn
from torchvision.transforms import transforms


class CNN(nn.Module):

    def __init__(self, network_params):
        super().__init__()
        self.__normalization = network_params["normalization"]

        conv_block_1 = network_params["layers"]["conv_block_1"]
        conv_block_2 = network_params["layers"]["conv_block_2"]
        conv_block_3 = network_params["layers"]["conv_block_3"]

        classifier = network_params["layers"]["classifier"]

        self.__cnn = nn.Sequential(
            # --- Conv block 1 ---
            self.__make_convolutional_block(conv_block_1),
            # --- Conv block 2 ---
            self.__make_convolutional_block(conv_block_2),
            # --- Conv block 3 ---
            self.__make_convolutional_block(conv_block_3),
            # --- Classifier ---
            nn.Flatten(),
            nn.Linear(**classifier["linear_1"]),
            nn.ReLU(),
            nn.Linear(**classifier["linear_2"]),
            nn.ReLU(),
            nn.Linear(in_features=classifier["linear_3"]["in_features"],
                      out_features=network_params["output_size"])
        )

    @staticmethod
    def __make_convolutional_block(block_data: Dict):
        return nn.Sequential(
            nn.Conv2d(**block_data["conv_2d_1"]),
            nn.ReLU(),
            nn.Conv2d(**block_data["conv_2d_2"]),
            nn.ReLU(),
            nn.MaxPool2d(**block_data["pool"]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, std = self.__normalization.values()
        normalization = transforms.Normalize(mean=mean, std=std)
        x = normalization(x)

        return self.__cnn(x)
