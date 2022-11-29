from typing import Dict

from classes.modules.base.CNN import CNN


class ImagesCNN(CNN):

    def __init__(self, network_params: Dict):
        img_size = (network_params["modality"]["size"]["width"], network_params["modality"]["size"]["height"])
        network_params["input_size"] = img_size
        network_params["layers"]["conv_block"]["conv_1"]["in_channels"] = network_params["modality"]["num_channels"]
        super().__init__(network_params)
