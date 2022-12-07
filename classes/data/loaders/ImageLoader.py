from typing import Dict, Callable

import torch
from PIL import Image
from torchvision.transforms import AutoAugment
from torchvision.transforms import transforms as T

# from torchvision.transforms.functional import InterpolationMode
from classes.data.loaders.Loader import Loader
from functional.yaml_manager import load_yaml


def apply_with_p(transformation, parameters: Dict) -> T.RandomApply:
    *params, (_, p) = parameters.items()
    return T.RandomApply([transformation(**dict(params))], p=p)


class ImageLoader(Loader):

    def __init__(self, img_details: Dict, transformations: Dict):
        super().__init__(img_details)
        self.__num_channels = img_details["num_channels"]
        self.__img_size = (img_details["size"]["width"], img_details["size"]["height"])
        self.__transformations = transformations
        if self.__transformations["manual"]:
            self._transforms = load_yaml(
                f"params/augmentations/{self.__transformations['transformations_preset']}.yml")

    def __get_transformations(self) -> Callable:
        """
        Creates a list of transformations to be applied to the inputs
        :return: a list of transformations to be applied to the inputs
        """
        # Automatic transformations following the AutoAugment approach (see paper)
        if self.__transformations["auto"]:
            return T.Compose([AutoAugment(), T.ToTensor()])
        # Manual transformations loaded from file.
        elif self.__transformations["manual"]:
            return T.Compose([
                T.ToTensor(),
                # Rotates an image with random angle
                apply_with_p(T.RandomRotation, self._transforms["rotation"]),
                # Performs random affine transform on an image
                apply_with_p(T.RandomAffine, self._transforms["random_affine"]),
                # Randomly transforms the morphology of objects in images and produces a see-through-water-like effect
                apply_with_p(T.ElasticTransform, self._transforms["elastic_transform"]),
                # Crops an image at a random location
                apply_with_p(T.RandomCrop, self._transforms["random_crop"]),
                # Randomly changes the brightness, saturation, and other properties of an image
                apply_with_p(T.ColorJitter, self._transforms["color_jitter"]),
                # Performs gaussian blur transform on an image
                apply_with_p(T.GaussianBlur, self._transforms["gaussian_blur"]),
                # Randomly selects a rectangle region in a torch Tensor image and erases its pixels (already has p)
                T.RandomErasing(**self._transforms["random_erasing"]),
                # Performs random perspective transform on an image
                apply_with_p(T.RandomPerspective, self._transforms["random_perspective"]),
            ])
        else:
            return T.ToTensor()

    def load(self, path_to_input: str) -> torch.Tensor:
        """
        Loads an image data item from the dataset
        :param path_to_input: the path to the data item to be loaded referred to the main modality
        :return: the image data item as a tensor
        """
        image = Image.open(path_to_input).convert('RGB')
        transformations = self.__get_transformations()
        # Note: if self.__num_channels = 3, it is the same as no indexing
        return transformations(image)[0:self.__num_channels, :, :]
