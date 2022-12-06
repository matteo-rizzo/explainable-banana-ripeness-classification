import logging
import os

import numpy as np
import skimage.segmentation as seg
from matplotlib import pyplot as plt
from skimage import io
from skimage.color import rgb2lab, deltaE_ciede2000
from skimage.morphology import convex_hull_image

from classes.core.Model import Model
from functional.segmentation import adjust_contrast, sharpen, blur_image


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

    def predict(self, x: np.ndarray, *args: any, **kwargs: any) -> np.ndarray:
        """
        Segment a batch of images
        :param x: 4D array of size (batch, height, width, channels)
        :return: batch of segmented images (same shape as input)
        """

        if not self.__parameters["check_clusters_order"]:
            logging.warning("check_clusters_order disabled. This could potentially create incorrect (inverted) masks, since no check on output is done.")

        # Contrast
        images = [adjust_contrast(img, **self.__parameters["contrast"]) for img in x]

        # Sharpening
        images = [sharpen(img, **self.__parameters["sharpen"]) for img in images]
        # (keep "amount" low, "radius" < 5)

        # Blur
        images = [blur_image(img, **self.__parameters["blur"]) for img in images]

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
        masks = np.stack([seg.slic(img, **kv) for img in images]).astype(bool)  # (batch, h, w)

        if self.__parameters["check_clusters_order"]:
            # SLIC uses k means clustering, but there is no guarantee that cluster 0 is background and 1 is banana. It could be inverted, and this
            # procedure ensures that bananas are always masked with 1, and background is at 0
            images = np.stack(images)
            neg_masks = ~masks
            # Get average RGB color in masked image (for inverted masks also)
            colors = np.where(masks.reshape(*masks.shape, 1), images, .0).sum(axis=(1, 2)) / masks.sum((1, 2)).reshape(-1, 1)  # (batch, rgb)
            neg_colors = np.where(neg_masks.reshape(*neg_masks.shape, 1), images, .0).sum(axis=(1, 2)) / neg_masks.sum((1, 2)).reshape(-1, 1)  # (batch, rgb)
            # Convert colors to LAB space for better comparisons
            lab_mask_colors = [(rgb2lab(c1), rgb2lab(c2)) for c1, c2 in zip(colors, neg_colors)]  # [(c1, c2), (color1, color2) ...] for each image
            # Color for comparison, since bananas are yellow
            yellow = rgb2lab([1.0, 1.0, 0.0])
            # For each image determine if mask or inverted mask should be used as real mask for bananas. This is determined looking in how close both colors are to yellow
            masks_idx = np.array([(deltaE_ciede2000(c1, yellow), deltaE_ciede2000(c2, yellow)) for c1, c2 in lab_mask_colors]).argmin(axis=1)  # (images,)
            masks_idx = np.eye(2)[masks_idx].astype(bool)
            masks = np.stack([masks, neg_masks], axis=1)[masks_idx]  # (batch, H, W)

        if self.__parameters["fill_contours"]:
            # masks = np.stack([binary_fill_holes(mask) for mask in masks]).astype(bool)  # (batch, H, W)
            masks = np.stack([convex_hull_image(mask) for mask in masks]).astype(bool)  # (batch, H, W)

        return x * masks[..., np.newaxis]


if __name__ == "__main__":
    import matplotlib as mpl

    mpl.rcParams["figure.dpi"] = 300
    imgs = [io.imread(os.path.join("_my_images", img_f), as_gray=False) for img_f in ["1.png", "2.png", "3.png", "4.png"]]
    model = ModelSLIC()
    msa = model.predict(np.stack(imgs))

    # check_clusters_order = False
    model = ModelSLIC(fill_contours=False)
    msb = model.predict(np.stack(imgs))

    fig, ax = plt.subplots(nrows=len(imgs), ncols=3, figsize=(10, 15))
    for i, (img, ma, mb) in enumerate(zip(imgs, msb, msa)):
        ax[i, 0].imshow(img)
        ax[i, 1].imshow(ma)
        ax[i, 2].imshow(mb)
        for a in ax:
            for b in a:
                b.axis("off")
    plt.tight_layout()
    plt.show()
