from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
from skimage import io, util
from skimage.color import label2rgb
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.feature import local_binary_pattern


def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)


def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')


def hist(ax, lbp):
    n_bins = int(lbp.max() + 1)
    return ax.hist(lbp.ravel(), density=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')


def _lbp(image, n_points, radius, METHOD):
    lbp = local_binary_pattern(image, n_points, radius, METHOD)

    # plot histograms of LBP of textures
    fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
    plt.gray()

    titles = ('edge', 'flat', 'corner')
    w = radius - 1
    edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
    flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
    i_14 = n_points // 4  # 1/4th of the histogram
    i_34 = 3 * (n_points // 4)  # 3/4th of the histogram
    corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                     list(range(i_34 - w, i_34 + w + 1)))

    label_sets = (edge_labels, flat_labels, corner_labels)

    for ax, labels in zip(ax_img, label_sets):
        ax.imshow(overlay_labels(image, lbp, labels))

    for ax, labels, name in zip(ax_hist, label_sets, titles):
        counts, _, bars = hist(ax, lbp)
        highlight_bars(bars, labels)
        ax.set_ylim(top=np.max(counts[:-1]))
        ax.set_xlim(right=n_points + 2)
        ax.set_title(name)

    ax_hist[0].set_ylabel('Percentage')
    for ax in ax_img:
        ax.axis('off')

    plt.show()


def main():
    # settings for LBP
    # radius = 3
    # n_points = 8 * radius
    #
    # # Params
    # METHOD = 'uniform'
    # plt.rcParams['font.size'] = 9
    #
    # image = io.imread("dataset/test_image/rank_0.png", as_gray=True)
    # _lbp(image, n_points, radius, METHOD)
    #
    # image = io.imread("dataset/test_image/rank_1.png", as_gray=True)
    # _lbp(image, n_points, radius, METHOD)
    #
    # image = io.imread("dataset/test_image/rank_2.png", as_gray=True)
    # _lbp(image, n_points, radius, METHOD)
    #
    # image = io.imread("dataset/test_image/rank_3.png", as_gray=True)
    # _lbp(image, n_points, radius, METHOD)

    image = io.imread("dataset/test_image/rank_3.png", as_gray=True)
    io.imshow(image)
    plt.show()

    image = util.invert(image)
    io.imshow(image)
    plt.show()

    blobs_log = blob_log(image, min_sigma=1, max_sigma=6, num_sigma=10, threshold=.1)

    # Compute radii in the 3rd column.
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    blobs_dog = blob_dog(image, min_sigma=1, max_sigma=6, threshold=.1)
    blobs_dog[:, 2] = blobs_dog[:, 2] * sqrt(2)

    blobs_doh = blob_doh(image, min_sigma=1, max_sigma=6, threshold=.01)

    blobs_list = [blobs_log, blobs_dog, blobs_doh]
    colors = ['yellow', 'lime', 'red']
    titles = ['Laplacian of Gaussian', 'Difference of Gaussian',
              'Determinant of Hessian']
    sequence = zip(blobs_list, colors, titles)

    fig, axes = plt.subplots(1, 3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    for idx, (blobs, color, title) in enumerate(sequence):
        ax[idx].set_title(title)
        ax[idx].imshow(image)
        for blob in blobs:
            y, x, r = blob
            c = plt.Circle((x, y), r, color=color, linewidth=2, fill=False)
            ax[idx].add_patch(c)
        ax[idx].set_axis_off()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
