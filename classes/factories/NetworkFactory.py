from typing import Dict

from torch import nn

from classes.modules.cnn_img.ImagesCNN import ImagesCNN
from classes.modules.pre_trained_cnn.PreTrainedCNN import PreTrainedCNN
from classes.modules.transformer.Transformer import Transformer


class NetworkFactory:
    networks_map = {
        "cnn_img": ImagesCNN,
        "pre_trained_cnn": PreTrainedCNN,
        "transformer": Transformer
    }

    def get(self, network_type: str, module_params: Dict, activation: bool = False) -> nn.Module:
        if network_type not in self.networks_map.keys():
            raise ValueError("Network {} is not implemented! \n Implemented networks are: {}"
                             .format(network_type, list(self.networks_map.keys())))
        return self.networks_map[network_type](module_params, activation)
