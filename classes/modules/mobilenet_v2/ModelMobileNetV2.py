from typing import Dict

from classes.core.Model import Model
from classes.modules.mobilenet_v2.MobileNetV2 import MobileNetV2


class ModelMobileNetV2(Model):

    def __init__(self, network_params: Dict):
        super().__init__(device=network_params["device"])
        self._network = MobileNetV2(network_params).to(self._device)
