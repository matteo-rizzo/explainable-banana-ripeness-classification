import logging
from array import array

import numpy as np
import skimage.segmentation as seg
from scipy.ndimage import binary_fill_holes
from skimage import exposure
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.filters import unsharp_mask, gaussian
from skimage.morphology import convex_hull_image

from classifiers.deep_learning.classes.core.Model import Model


class ModelSLIC(Model):
    _default_parameters = {
        "contrast": dict(p_min=1.0, p_max=99.0),
        "sharpen": dict(radius=4.0, amount=1.0),
        "blur": dict(std=1.0),
        "check_clusters_order": True,
        "fill_contours": True
    }

    def __init__(self, device=None, **kwargs):
        super().__init__(device)
        self.__parameters = ModelSLIC._default_parameters
        self.__parameters.update(kwargs)

    @staticmethod
    def __adjust_contrast(img, p_min=1.0, p_max=99.0) -> array.pyi:
        vmin, vmax = np.percentile(img, q=(p_min, p_max))
        return exposure.rescale_intensity(img, in_range=(vmin, vmax), out_range=np.float32)

    @staticmethod
    def __sharpen(img, radius: float = 1.0, amount: float = 1.0) -> np.ndarray:
        return unsharp_mask(img, radius=radius, amount=amount, preserve_range=False, channel_axis=None)

    @staticmethod
    def __blur_image(img, std: float = 1.0) -> np.ndarray:
        return gaussian(img, sigma=std, channel_axis=-1)

    def predict(self, x: np.ndarray, convex_hull: bool = False, *args: any, **kwargs: any) -> np.ndarray:
        """
        Segment a batch of images
        :param x: 4D array of size (batch, height, width, channels)
        :param convex_hull: whether to compute the convex hull of the segmentation mask
        :return: batch of segmented images (same shape as input)
        """

        if not self.__parameters["check_clusters_order"]:
            logging.warning("check_clusters_order disabled. This could potentially create incorrect (inverted) masks, "
                            "since no check on output is done.")

        # Contrast
        images = [self.__adjust_contrast(img, **self.__parameters["contrast"]) for img in x]

        # Sharpening
        images = [self.__sharpen(img, **self.__parameters["sharpen"]) for img in images]
        # (keep "amount" low, "radius" < 5)

        # Blur
        images = [self.__blur_image(img, **self.__parameters["blur"]) for img in images]

        # SLIC clustering parameters
        kv = dict(n_segments=2,  # ok, background/banana
                  start_label=0,  # 0 is background, 1 is banana
                  max_num_iter=5,  # tradeoff for speed
                  slic_zero=False,  # leave
                  compactness=10.0,
                  # "compactness" should be tuned, higher values give more weight to space proximity,
                  # making superpixel shapes more square/cubic.
                  # Does not seem to make any difference with values <= 10. Higher values works worse
                  enforce_connectivity=False,
                  # if True better contours, but some non-banana regions.
                  # If False follows bananas better, but black spots are sometimes left out. More problematic
                  max_size_factor=3,  # can be tuned
                  convert2lab=True,  # leave as is
                  channel_axis=-1)  # leave as is

        # Predict numeric label 0/1 for each pixel of the image
        masks = np.stack([seg.slic(img, **kv) for img in images]).astype(bool)  # (batch, h, w)

        if self.__parameters["check_clusters_order"]:
            # SLIC uses k means clustering, but there is no guarantee that cluster 0 is background and 1 is banana.
            # It could be inverted, and this procedure ensures bananas are always masked with 1, and background is at 0
            images = np.stack(images)
            neg_masks = ~masks
            # Get average RGB color in masked image (for inverted masks also)
            colors = np.where(masks.reshape(*masks.shape, 1), images, .0).sum(axis=(1, 2)) / masks.sum((1, 2)).reshape(
                -1, 1)  # (batch, rgb)
            neg_colors = np.where(neg_masks.reshape(*neg_masks.shape, 1), images, .0).sum(axis=(1, 2)) / neg_masks.sum(
                (1, 2)).reshape(-1, 1)  # (batch, rgb)
            # Convert colors to LAB space for better comparisons [(c1, c2), (color1, color2) ...] for each image
            lab_mask_colors = [(rgb2lab(c1), rgb2lab(c2)) for c1, c2 in zip(colors, neg_colors)]
            # Color for comparison, since bananas are yellow
            yellow = rgb2lab([1.0, 1.0, 0.0])
            # For each image determine if mask or inverted mask should be used as real mask for bananas.
            # This is determined looking in how close both colors are to yellow
            masks_idx = np.array(
                [(deltaE_ciede2000(c1, yellow), deltaE_ciede2000(c2, yellow)) for c1, c2 in lab_mask_colors]).argmin(
                axis=1)  # (images,)
            masks_idx = np.eye(2)[masks_idx].astype(bool)
            masks = np.stack([masks, neg_masks], axis=1)[masks_idx]  # (batch, H, W)

        if self.__parameters["fill_contours"]:
            if convex_hull:
                masks = np.stack([convex_hull_image(mask) for mask in masks]).astype(bool)  # (batch, H, W)
            else:
                masks = np.stack([binary_fill_holes(mask) for mask in masks]).astype(bool)  # (batch, H, W)

        return x * masks[..., np.newaxis]
