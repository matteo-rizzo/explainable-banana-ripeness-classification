from os import PathLike
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
from lime import lime_image
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from skimage.segmentation import mark_boundaries
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from classes.explainability.InterpretabilityModel import InterpretabilityModel


def get_preprocess_transform():
    # TODO: use correct values
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    return transf


def batch_predict(images, model: nn.Module):
    model.eval()
    preprocess = get_preprocess_transform()
    batch = torch.stack(tuple(preprocess(i) for i in images), dim=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    batch = batch.to(device)

    logits = model(batch)
    probs = F.softmax(logits, dim=1)
    return probs.detach().cpu().numpy()


def create_figure(n_rows: int = 1, n_cols: int = 1) -> Tuple[Figure, np.ndarray]:
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 3, n_rows * 3))
    if isinstance(axs, list):
        for a in axs:
            a.axis("off")
    else:
        for i in range(n_rows):
            for j in range(n_cols):
                axs[i, j].axis("off")
    return fig, axs


class ModelLIME(InterpretabilityModel):
    def __init__(self, model, device: torch.device, save_path: PathLike, num_train_images: Optional[int] = 32, num_test_images: Optional[int] = 5):
        super().__init__(model)
        self._device = device
        self._num_train_images = num_train_images
        self._num_test_images = num_test_images
        self._save_path = Path(save_path)

    def explain(self, test_loader: DataLoader, train_loader: DataLoader, label_names: List) -> None:
        test_images, true_labels = ModelLIME.get_batch_from_loader(test_loader, classes=label_names, num_per_class=1)

        explainer = lime_image.LimeImageExplainer()

        f, axs = create_figure(test_images.size(0), len(label_names) + 1)

        # transformations = get_pil_transform()
        for i, img in enumerate(test_images):
            img = (img.transpose(0, -1).detach().cpu().numpy() * 255).astype(np.uint8)
            explanation = explainer.explain_instance(img,
                                                     lambda data: batch_predict(data, model=self._model),  # classification function
                                                     labels=label_names,
                                                     top_labels=len(label_names),
                                                     hide_color=0,
                                                     num_samples=1000)  # number of images that will be sent to classification function
            # num_samples should be higher, 100

            pred_labels = explanation.top_labels
            axs[i, 0].imshow(img)

            for j, lab in enumerate(pred_labels):
                temp, mask = explanation.get_image_and_mask(lab, positive_only=True, num_features=5, hide_rest=True)
                image_boundary_more = mark_boundaries(temp / 255.0, mask)

                axs[i, j + 1].imshow(image_boundary_more)
                axs[i, j + 1].set_title(f"{lab}")

        f.suptitle("LIME output", fontsize=16)
        self._save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(self._save_path / "LIME_ImageExplainer.png", dpi=500, bbox_inches="tight")