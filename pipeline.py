import os.path
import time
from typing import Tuple

import matplotlib as mpl
import numpy as np
import skimage.segmentation as seg
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from skimage import io, color
from skimage.transform import resize

from functional.segmentation import adjust_contrast, sharpen, blur_image

mpl.rcParams["figure.dpi"] = 300

# base_images = os.path.join("dataset", "treviso-market-224_224", "0")
base_images = os.path.join("_my_images")

IMAGES = [os.path.join(base_images, s) for s in ["2.png", "1.png"]]


def rescale_img(img: np.ndarray, scaled_w: int = 512, **kwargs) -> np.ndarray:
    h, w, *c = img.shape
    scaled_h = h * scaled_w // w

    img = resize(img, (scaled_h, scaled_w), **kwargs)
    return img


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


def main():
    # Load images
    imgs = [io.imread(img_f, as_gray=False) for img_f in IMAGES]

    # Rescale -> w=512
    imgs_proc = [rescale_img(img, scaled_w=512) for img in imgs]
    # imgs_proc = imgs

    start_img = time.perf_counter()
    # Contrast
    imgs_proc = [adjust_contrast(img, p_min=1.0, p_max=99.0) for img in imgs_proc]

    # Sharpening
    # imgs_proc = [sharpen(img, radius=4.0, amount=1.0) for img in imgs_proc]
    # # (keep "amount" low, "radius" < 5)
    #
    # # Blur
    # imgs_proc = [blur_image(img, std=1.0) for img in imgs_proc]

    # SLIC clustering
    kv = dict(n_segments=2,  # ok, background/banana
              start_label=0,  # 0 is background, 1 is banana
              # max_num_iter=30,  # tradeoff for speed
              # slic_zero=False,  # leave
              # compactness=10.0,
              # # "compactness" should be tuned, higher values give more weight to space proximity, making superpixel shapes more square/cubic
              # # Does not seem to make any difference with values <= 10. Higher values works worse
              enforce_connectivity=True,
              # # if True better contours, but some non-banana regions.
              # # If False follows bananas better, but black spots are sometimes left out. More problematic
              convert2lab=True,  # leave as is
              channel_axis=-1)  # leave as is

    start = time.perf_counter()
    # Predict numeric labels [0-18] for each pixel of the image
    masks = [seg.slic(img, **kv) for img in imgs_proc]
    # masks = seg.slic(np.stack(imgs_proc), **kv)
    end = time.perf_counter()
    masks = [rescale_img(m, oi.shape[1], order=0, preserve_range=True) for m, oi in zip(masks, imgs)]
    print(f"SLIC time per image: {(end - start) / len(IMAGES)} s;   TOTAL: {(end - start_img) / len(IMAGES)} s")

    # PLOT results
    f, axs = create_figure(len(IMAGES), 4)
    f.suptitle("sharpening", fontsize=16)
    for i in range(len(IMAGES)):
        img, img_proc, mask = imgs[i], imgs_proc[i], masks[i]
        axs[i, 0].imshow(img)
        axs[i, 1].imshow(img_proc)
        axs[i, 2].imshow(color.label2rgb(mask, img, kind="overlay"))
        axs[i, 3].imshow(img * mask[..., np.newaxis])
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
