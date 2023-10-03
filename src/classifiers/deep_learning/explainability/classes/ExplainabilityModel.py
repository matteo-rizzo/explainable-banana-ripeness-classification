from collections import Counter
from typing import List, Tuple, Optional

import torch.utils.data
from torch import nn
from torch.utils.data import DataLoader


class ExplainabilityModel:
    def __init__(self, model: nn.Module):
        self._model: nn.Module = model

    @staticmethod
    def get_batch_from_loader(loader: DataLoader, num_images: int = None, classes: List[int] = None,
                              num_per_class: int = None) -> Tuple[torch.Tensor, Optional[List]]:
        """
        Extract a number of images from a torch DataLoader and stack them together in a 4D tensor

        :param loader: dataset loader
        :param num_images: max number of images
        :param classes: if provided specifies that "num_per_class" images should be selected for each class. If None, random images are used
        :param num_per_class: number of images to be selected for each class (if None 1 is default)
        :return: tensor of images, with list of labels iff "classes" parameters was specified
        """

        if classes is not None:
            if num_per_class is None:
                num_per_class = 1
        elif num_images is None:
            raise ValueError("Either 'num_images' or 'classes' must be provided.")

        if classes is None:
            if loader.batch_size >= num_images:
                batch = next(iter(loader))
                images, _ = batch
            else:
                images = list()
                for batch in loader:
                    imgs, _ = batch
                    images.append(imgs)
                    if len(images) >= num_images:
                        break
                if len(images) < num_images:
                    raise ValueError(f"You asked for {num_images} images."
                                     f" DataLoader contained only {len(images)}. Check content of DataLoader")
                images = torch.cat(images)
            images = images[:num_images, ...], None
        else:  # per class add
            done = False
            images = list()
            labels = list()
            count_classes = Counter()
            for batch in loader:
                if done:
                    break
                imgs, ys = batch
                for img, y in zip(imgs, ys):
                    if done:
                        break
                    y = y.item()
                    if count_classes.get(y, 0) < num_per_class:
                        images.append(img)
                        labels.append(y)
                        count_classes.update({y: 1})
                    if set(classes) == set(count_classes.keys()) and all([v == num_per_class for v in count_classes.values()]):
                        done = True

            if not done:
                raise Exception(f"You asked for {num_per_class} images for each class but not enough images were found. Check content of DataLoader")
            images = torch.stack(images), labels
        return images

    def explain(self, test_loader: DataLoader, train_loader: DataLoader, label_names: Optional[List]) -> None:
        """
        Generate plots to explain the test images

        :param test_loader: loader with test set images
        :param train_loader: training set of images
        :param label_names: optional label names
        """
        pass
