from typing import Dict

from classes.core.Model import Model
from classes.modules.pre_trainet_vit.PreTrainedViT import PreTrainedViT


class ModelPreTrainedCNN(Model):

    def __init__(self, network_params: Dict):
        super().__init__(device=network_params["device"])
        self._network = PreTrainedViT(network_params).to(self._device)
