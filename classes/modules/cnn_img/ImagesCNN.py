from typing import Dict

from torch import nn


class ImagesCNN(nn.Module):

    def __init__(self, network_params: Dict, activation: bool = True):
        img_size = (network_params["modality"]["size"]["width"], network_params["modality"]["size"]["height"])
        network_params["input_size"] = img_size
        network_params["layers"]["conv_block"]["conv_1"]["in_channels"] = network_params["modality"]["num_channels"]
        super().__init__()
