from typing import Dict

from classes.modules.transformer.Transformer import Transformer
from torch import nn

from classes.modules.cnn.CNN import CNN
from classes.modules.mobilenet_v2.MobileNetV2 import MobileNetV2
from classes.modules.pre_trained_vit.PreTrainedViT import PreTrainedViT


class NetworkFactory:
    networks_map = {
        "cnn": CNN,
        "mobilenet_v2": MobileNetV2,
        "transformer": Transformer,
        "pre_trained_vit": PreTrainedViT
    }

    def get(self, network_type: str, module_params: Dict, activation: bool = False) -> nn.Module:
        if network_type not in self.networks_map.keys():
            raise ValueError(
                f"Network {network_type} is not implemented! "
                f"\n Implemented networks are: {list(self.networks_map.keys())}")
        return self.networks_map[network_type](module_params, activation)
