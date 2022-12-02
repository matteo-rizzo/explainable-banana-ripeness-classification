import torch.utils.data
from torch.utils.data import DataLoader

from classes.core.Model import Model


class InterpretabilityModel:
    def __init__(self, model: Model):
        self._model = model

    @staticmethod
    def get_batch_from_loader(loader: DataLoader, num_images: int) -> torch.Tensor:
        """
        Extract a number of images from a torch DataLoader and stack them together in a 4D tensor

        :param loader: dataset loader
        :param num_images: max
        :return:
        """
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
                raise Exception(f"You asked for {num_images} images DataLoader contained only {len(images)}. Check content of DataLoader")
            images = torch.cat(images)
        return images[:num_images, ...]

    def explain(self, test_loader: DataLoader, train_loader: DataLoader) -> None:
        """
        Generate plots to explain the test images

        :param test_loader: loader with test set images
        :param train_loader: training set of images
        """
        pass
