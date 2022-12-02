from os import PathLike
from pathlib import Path
from typing import Optional, List

import numpy as np
import shap
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from classes.explainability.InterpretabilityModel import InterpretabilityModel


class ModelSHAP(InterpretabilityModel):
    def __init__(self, model, device: torch.device, save_path: PathLike, num_train_images: Optional[int] = 32, num_test_images: Optional[int] = 5):
        super().__init__(model)
        self._device = device
        self._num_train_images = num_train_images
        self._num_test_images = num_test_images
        self._save_path = Path(save_path)

    def explain(self, train_loader: DataLoader, test_loader: DataLoader, label_names: List) -> None:
        """
        Takes train and test images and fit a DeepExplainer to show heatmaps over a subset of test images

        :param train_loader: the loader containing train images
        :param test_loader: the loader containing test images
        :param label_names: list of names for each output class
        """

        num_class = self._num_train_images // len(label_names)

        train_images, _ = ModelSHAP.get_batch_from_loader(train_loader, classes=label_names, num_per_class=num_class)
        test_images, true_labels = ModelSHAP.get_batch_from_loader(test_loader, classes=label_names, num_per_class=1)

        background = train_images.to(self._device)
        test_images = test_images.to(self._device)

        e = shap.DeepExplainer(self._model, background)
        shap_values = e.shap_values(test_images)

        if label_names is not None:
            label_names = np.array(label_names).reshape(1, -1).repeat(self._num_test_images, axis=0)

        # Issue in: https://github.com/slundberg/shap/issues/703
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
        test_numpy = np.swapaxes(np.swapaxes(test_images.cpu().numpy(), 1, -1), 1, 2)
        shap.image_plot(shap_numpy, test_numpy, show=False, labels=label_names)  # was -test_numpy

        # Save figure
        self._save_path.mkdir(parents=True, exist_ok=True)
        # plt.gcf().axes[-1].set_aspect("auto")
        # plt.tight_layout()

        plt.savefig(self._save_path / "SHAP_DeepExplainer.png", dpi=500, bbox_inches="tight")
