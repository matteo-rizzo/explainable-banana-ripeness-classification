from typing import Dict
from typing import List

import torch
from PIL import Image
from torchvision.transforms import transforms

# from torchvision.transforms.functional import InterpolationMode
from classes.data.loaders.Loader import Loader


class ImageLoader(Loader):

    def __init__(self, img_details: Dict):
        super().__init__(img_details)
        self.__num_channels = img_details["num_channels"]
        self.__img_size = (img_details["size"]["width"], img_details["size"]["height"])

    @staticmethod
    def __get_transformations() -> List:
        """
        Creates a list of transformations to be applied to the inputs
        :return: a list of transformations to be applied to the inputs
        """
        return [
            # transforms.Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor()]

        # Note: Normalizing here is wasteful; tensor is still on CPU
        # return [transforms.Resize(self.__img_size, interpolation=InterpolationMode.BILINEAR),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             mean=(0.485, 0.456, 0.406),
        #             std=(0.229, 0.224, 0.225))]
        # mean = (0.5, 0.5, 0.5),
        # std = (0.5, 0.5, 0.5)

    def load(self, path_to_input: str) -> torch.Tensor:
        """
        Loads an image data item from the dataset
        :param path_to_input: the path to the data item to be loaded referred to the main modality
        :return: the image data item as a tensor
        """
        image = Image.open(path_to_input).convert('RGB')
        transformations = self.__get_transformations()
        # Note: if self.__num_channels = 3, it is the same as no indexing
        return transforms.Compose(transformations)(image)[0:self.__num_channels, :, :]
