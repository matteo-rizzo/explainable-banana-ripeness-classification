import time

import skimage.color as color
import skimage.segmentation as seg
from matplotlib import pyplot as plt
from skimage import io
from skimage.transform import resize


def rescale(img, scaled_w: int = 512):
    h, w, c = img.shape

    scaled_h = h * scaled_w // w

    img = resize(img, (scaled_h, scaled_w), anti_aliasing=True)
    return img


def image_show(image, nrows=1, ncols=1, cmap="gray"):
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(14, 14))
    ax[0].imshow(image, cmap="gray")
    ax[0].axis("off")
    return fig, ax


image = io.imread("segmentation/image.png", as_gray=False)
image = rescale(image)
# image_gray = color.rgb2gray(image)
f, ax = image_show(image, ncols=2)

start = time.perf_counter()
# Predict numeric labels [0-18] for each pixel of the image
image_slic = seg.slic(image, n_segments=2, max_num_iter=30)
end = time.perf_counter()
print(f"Time: {end - start} s")

ax[1].imshow(color.label2rgb(image_slic, image, kind="overlay"))
plt.show()

# image_felzenszwalb = seg.felzenszwalb(image)
# image_felzenszwalb_colored = color.label2rgb(image_felzenszwalb, image, kind="avg")
# image_show(image_felzenszwalb_colored);
