import time

import matplotlib as mpl
import skimage.color as color
import skimage.segmentation as seg
from matplotlib import pyplot as plt
from skimage import io
from skimage.transform import resize

mpl.rcParams['figure.dpi'] = 300


def rescale(img, scaled_w: int = 512, **kwargs):
    h, w, *c = img.shape
    scaled_h = h * scaled_w // w

    img = resize(img, (scaled_h, scaled_w), **kwargs)
    return img


def image_show(image, nrows=1, ncols=1, cmap="gray"):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5))
    ax[0].imshow(image, cmap="gray")
    for a in ax:
        a.axis("off")
    return fig, ax


if __name__ == "__main__":
    image_o = io.imread("segmentation/image.png", as_gray=False)
    h, w, c = image_o.shape
    image = rescale(image_o, anti_aliasing=True)
    # image = color.rgb2gray(image)
    f, ax = image_show(image, ncols=2)

    start = time.perf_counter()
    # Predict numeric labels [0-18] for each pixel of the image
    mask = seg.slic(image,
                    n_segments=2,  # ok
                    start_label=0,  # 0 is background, 1 is banana
                    max_num_iter=10,  # tradeoff for speed
                    slic_zero=False,  # leave
                    compactness=10.0,  # should be tuned
                    enforce_connectivity=True,
                    # if True better contours, but some non-banana regions.
                    # If False follows bananas better, but black spots are sometimes left out. More problematic
                    channel_axis=-1)  # leave as is
    end = time.perf_counter()
    print(f"Time: {end - start} s")

    mask_r = rescale(mask, w, order=0, preserve_range=True)

    ax[1].imshow(color.label2rgb(mask_r, image_o, kind="overlay"))
    plt.show()

    # image_felzenszwalb = seg.quickshift(image, ratio=1.0, kernel_size=100)
    # ax[1].imshow(color.label2rgb(image_felzenszwalb, image, kind="overlay"))
    # plt.show()
