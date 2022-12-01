import torch.utils.data

from classes.core.Model import Model


class InterpretabilityModel:
    def __init__(self, model: Model):
        self._model = model

    def explain(self, loader: torch.utils.data.DataLoader) -> None:
        pass
