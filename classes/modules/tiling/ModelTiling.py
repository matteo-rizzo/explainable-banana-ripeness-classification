from typing import Dict

import torch
from torch.nn import Unfold

from classes.core.Model import Model


class ModelTiling(Model):
    _default_parameters = dict(kernel_size=(64, 64), dilation=1, padding=0, stride=64, device="cuda")

    def __init__(self, network_params: Dict = _default_parameters):
        super().__init__(device=network_params.pop("device"))
        self._network = Unfold(**network_params).to(self._device)

    def predict(self, x: torch.Tensor, *args: any, **kwargs: any) -> torch.Tensor:
        """
        Tile a batch of 2D images into patches

        :param x: 4D tensor (batch, channels, height, width)
        :param args:
        :param kwargs:
        :return: 3D tensor (batch, features, n_patches). "features" is given by (kernel_size * channels)
        """

        return self._network(x)

# Example of usage
# if __name__ == '__main__':
#     m = ModelTiling()
#     # m = ModelSLIC()
#
#     x = torch.randn((8, 3, 128, 128))
#
#     # xs = m.predict(x.transpose(1, -1).numpy())
#     xs = m.predict()
#     pass
