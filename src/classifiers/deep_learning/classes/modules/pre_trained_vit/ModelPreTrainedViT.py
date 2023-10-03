from typing import Dict

from src.classifiers.deep_learning.classes.core.Model import Model
from src.classifiers.deep_learning.classes.modules.pre_trained_vit.PreTrainedViT import PreTrainedViT


class ModelPreTrainedVit(Model):

    def __init__(self, network_params: Dict):
        super().__init__(device=network_params["device"])
        self._network = PreTrainedViT(network_params).to(self._device)
