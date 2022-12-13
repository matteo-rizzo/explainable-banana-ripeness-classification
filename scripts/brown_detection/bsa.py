import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2lab, rgba2rgb
from skimage.morphology import dilation, erosion, opening


def rgba_to_scaled_lab(image_rgba):
    image_lab = rgb2lab(rgba2rgb(image_rgba))
    # Rescale LAB images because of skimage shenanigans.
    # This was actually the solution of one of the authors
    lab_scaled = (image_lab + [0, 128, 128]) / [100, 255, 255]
    return lab_scaled


def binarize_lab(image_lab):
    threshold = 130 / 255
    l, a, b = image_lab[:, :, 0], image_lab[:, :, 1], image_lab[:, :, 2]
    binarized_image = (a > threshold).astype(int)

    return binarized_image


def main():
    images = [io.imread("dataset/test_image/rank_0.png"),
              io.imread("dataset/test_image/rank_1.png"),
              io.imread("dataset/test_image/rank_2.png"),
              io.imread("dataset/test_image/rank_3.png"),
              ]

    fig, axes = plt.subplots(4, 4, figsize=(9, 9))

    for ax, img in zip(axes, images):
        ax_1, ax_2, ax_3, ax_4 = ax
        # ---------------------------
        for x in ax:
            x.axis('off')
        # -------- Base image -------
        ax_1.set_title("Base image")
        ax_1.imshow(img)
        # -------- LAB image --------
        ax_2.set_title("LAB image")
        lab = rgba_to_scaled_lab(img)
        ax_2.imshow(lab)
        # ---- Binary mask image ----
        ax_3.set_title("Masked image")
        bin_msk = binarize_lab(lab)
        ax_3.imshow(bin_msk)
        # --- Morphed mask image ----
        ax_4.set_title("Morphed mask")
        morphed = opening(bin_msk)
        ax_4.imshow(morphed)
        # ---------------------------
        print(f"{((morphed.sum() / morphed.size) * 100):2f}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
