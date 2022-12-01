import torch

from classes.explainability.InterpretabilityModel import InterpretabilityModel


class ModelLIME(InterpretabilityModel):
    def __init__(self, model, device: torch.device):
        super().__init__(model)
        self._device = device

    def explain(self, loader: torch.utils.data.DataLoader) -> None:
        """
        Takes batch of images > 105 and use the last 5 to get explanation

        :param loader: the loader containing test images
        """

        pass
