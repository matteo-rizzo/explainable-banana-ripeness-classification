from typing import Dict

from torch import nn

from classes.modules.cnn_img.ImagesCNN import ImagesCNN
from classes.modules.pre_trained_cnn.PreTrainedCNN import PreTrainedCNN
from classes.modules.transformer.Transformer import Transformer


class NetworkFactory:
    networks_map = {
        "cnn": ImagesCNN,
        "pre_trained_cnn": PreTrainedCNN,
        "transformer": Transformer
    }

    def get(self, network_type: str, module_params: Dict, activation: bool = False) -> nn.Module:
        if network_type not in self.networks_map.keys():
            raise ValueError(
                f"Network {network_type} is not implemented! "
                f"\n Implemented networks are: {list(self.networks_map.keys())}")
        return self.networks_map[network_type](module_params, activation)
