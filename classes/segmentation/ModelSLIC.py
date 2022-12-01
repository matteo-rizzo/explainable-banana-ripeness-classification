from typing import Dict

import numpy as np
import skimage.segmentation as seg

from classes.core.Model import Model
from classes.segmentation.utils import sharpen, adjust_contrast, blur_image


class ModelSLIC(Model):
    _default_parameters = {
        "contrast": dict(p_min=1.0, p_max=99.0),
        "sharpen": dict(radius=4.0, amount=1.0),
        "blur": dict(std=1.0)
    }

    def __init__(self, params: Dict = _default_parameters, device=None):
        super().__init__(device)
        self.__parameters = params

    def predict(self, x: np.ndarray, *args: any, **kwargs: any) -> np.ndarray:
        """
        Segment a batch of images

        :param x: 4D array of size (batch, height, width, channels)
        :return: batch of segmented images (same shape as input)
        """

        # Contrast
        imgs_proc = [adjust_contrast(img, **self.__parameters["contrast"]) for img in x]

        # Sharpening
        imgs_proc = [sharpen(img, **self.__parameters["sharpen"]) for img in imgs_proc]
        # (keep "amount" low, "radius" < 5)

        # Blur
        imgs_proc = [blur_image(img, **self.__parameters["blur"]) for img in imgs_proc]

        # SLIC clustering parameters
        kv = dict(n_segments=2,  # ok, background/banana
                  start_label=0,  # 0 is background, 1 is banana
                  max_num_iter=5,  # tradeoff for speed
                  slic_zero=False,  # leave
                  compactness=10.0,
                  # "compactness" should be tuned, higher values give more weight to space proximity, making superpixel shapes more square/cubic
                  # Does not seem to make any difference with values <= 10. Higher values works worse
                  enforce_connectivity=True,
                  # if True better contours, but some non-banana regions.
                  # If False follows bananas better, but black spots are sometimes left out. More problematic
                  max_size_factor=3,  # can be tuned
                  convert2lab=True,  # leave as is
                  channel_axis=-1)  # leave as is

        # Predict numeric label 0/1 for each pixel of the image
        masks = np.stack([seg.slic(img, **kv) for img in imgs_proc])  # (batch, h, w)
        return x * masks[..., np.newaxis]
