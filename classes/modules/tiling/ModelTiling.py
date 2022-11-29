from typing import Dict

import torch
from torch.nn import Unfold

from classes.core.Model import Model


class ModelTiling(Model):
    """
    Image tile extractor: extract patches from batch of images and stack them together.
    Look at https://pytorch.org/docs/stable/generated/torch.nn.Unfold.html for a list of accepted "network_params".

    The "resize" parameter controls whether patches are stacked as 3D tensors (true) or flattened in a 1D tensor (false).
    Look at the :py:meth:`.ModelTiling.predict`.
    """
    _default_parameters = dict(kernel_size=(64, 64), dilation=1, padding=0, stride=64)

    def __init__(self, device: torch.device,
                 network_params: Dict = _default_parameters,
                 resize: bool = False):
        super().__init__(device=device)
        self._network = Unfold(**network_params).to(self._device)
        self.__resize: bool = resize

    def predict(self, x: torch.Tensor, *args: any, **kwargs: any) -> torch.Tensor:
        """
        Tile a batch of 2D images into patches

        :param x: 4D tensor (batch, channels, height, width)
        :return: If "resize" is False a 3D tensor (batch, n_patches, features). "features" is given by (patch_height * patch_width * channels)
            If "resize" is True a 5D tensor as (batch, n_patches, channels, patch_height, patch_width)
        """

        # Batch and channels num are preserved in output
        b, c, *_ = x.size()
        # Get the patches dimensions
        ph, pw = self._network.kernel_size
        # Extract patches
        o = self._network(x).transpose(1, 2)
        # Reshape if needed
        if self.__resize:
            return o.reshape(b, -1, c, ph, pw)
        return o

# Example of usage
# if __name__ == '__main__':
#     m = ModelTiling(device="cuda", resize=True)
#     # m = ModelSLIC()
#
#     x = torch.randn((8, 3, 256, 256))
#
#     # xs = m.predict(x.transpose(1, -1).numpy())
#     xs = m.predict(x)
#     pass
