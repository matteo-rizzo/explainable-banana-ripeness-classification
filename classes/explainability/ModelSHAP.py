import numpy as np
import shap
import torch
from torch.utils.data import DataLoader

from classes.explainability.InterpretabilityModel import InterpretabilityModel


class ModelSHAP(InterpretabilityModel):
    def __init__(self, model, device: torch.device, num_train_images: int = 32, num_test_images: int = 5):
        super().__init__(model)
        self._device = device
        self._num_train_images = num_train_images
        self._num_test_images = num_test_images

    def explain(self, loader: DataLoader) -> None:
        """
        Takes batch of images > 105 and use the last 5 to get explanation.

        :param loader: the loader containing test images
        """

        if loader.batch_size > self._num_train_images + self._num_test_images:
            batch = next(iter(loader))
            images, _ = batch
        else:
            images = list()
            for batch in loader:
                imgs, _ = batch
                images.append(imgs)
                if len(images) > self._num_train_images + self._num_test_images:
                    break
            images = torch.cat(images)

        background = images[:self._num_train_images].to(self._device)
        test_images = images[self._num_train_images:self._num_train_images + self._num_test_images].to(self._device)

        e = shap.DeepExplainer(self._model, background)
        shap_values = e.shap_values(test_images)

        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)
        shap.image_plot(shap_numpy, -test_numpy)
        # shap.summary_plot(shap_values, test_images, feature_names)
