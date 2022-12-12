import matplotlib.pyplot as plt
from skimage import io
from skimage.color import rgb2lab, rgba2rgb
from skimage.morphology import dilation, erosion, opening


def lab_binarize(image, ax):
    image_lab = rgb2lab(rgba2rgb(image))
    # Rescale LAB images because of skimage shenanigans.
    # This was actually the solution of one of the authors
    lab_scaled = (image_lab + [0, 128, 128]) / [100, 255, 255]
    # io.imshow(lab_scaled)
    # plt.show()
    threshold = 130 / 255
    l, a, b = lab_scaled[:, :, 0], lab_scaled[:, :, 1], lab_scaled[:, :, 2]
    binarized_a = (a > threshold).astype(int)

    ax.imshow(binarized_a)
    return binarized_a


def show_binarized_brownies():
    images = [io.imread("dataset/test_image/rank_0.png"),
              io.imread("dataset/test_image/rank_1.png"),
              io.imread("dataset/test_image/rank_2.png"),
              io.imread("dataset/test_image/rank_3.png"),
              ]
    # Binarized images based on brownnnness
    binarized = []
    fig, axx = plt.subplots(4, 2, figsize=(9, 9))
    for ax, img in zip(axx, images):
        ax_left, ax_right = ax
        ax_left.axis('off')
        ax_right.axis('off')

        ax_left.imshow(img)
        binarized.append(lab_binarize(img, ax_right))
    plt.tight_layout()
    plt.show()

    return binarized


def apply_morphology(binarized):
    fig, axx = plt.subplots(4, 5, figsize=(9, 9))

    for ax, binarized_img in zip(axx, binarized):
        ax_1, ax_2, ax_3, ax_4, ax_5 = ax
        for x in ax:
            x.axis('off')

        ax_1.set_title("brown mask")
        ax_1.imshow(binarized_img)

        im1 = opening(binarized_img)
        ax_2.set_title("opening")
        ax_2.imshow(im1)

        im2 = erosion(binarized_img)
        ax_3.set_title("erosion")
        ax_3.imshow(im2)

        im3 = dilation(binarized_img)
        ax_4.set_title("dilation")
        ax_4.imshow(im3)

        im4 = dilation(erosion(opening(binarized_img)))
        ax_5.set_title("all three")
        ax_5.imshow(im4)
    plt.show()


def main():
    binarized = show_binarized_brownies()
    apply_morphology(binarized)


if __name__ == "__main__":
    main()
