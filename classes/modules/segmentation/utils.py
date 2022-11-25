"""
Image manipulation utilities to aid the segmentation process
"""

import numpy as np
from skimage import exposure
from skimage.filters import unsharp_mask, gaussian


def adjust_contrast(img, p_min=1.0, p_max=99.0) -> np.ndarray:
    vmin, vmax = np.percentile(img, q=(p_min, p_max))
    img = exposure.rescale_intensity(img,
                                     in_range=(vmin, vmax),
                                     out_range=np.float32,
                                     )
    return img


def sharpen(img, radius: float = 1.0, amount: float = 1.0) -> np.ndarray:
    return unsharp_mask(img, radius=radius, amount=amount,
                        preserve_range=False,
                        channel_axis=None,  # -1 does not work
                        )


def blur_image(img, std: float = 1.0) -> np.ndarray:
    return gaussian(img, sigma=std, channel_axis=-1)
