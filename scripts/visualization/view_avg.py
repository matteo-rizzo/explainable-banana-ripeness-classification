from math import sqrt, floor, ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

data = ["dataset/treviso-market-224_224-RGB.csv",
        "dataset/treviso-market-224_224-seg-RGB.csv",
        "dataset/treviso-market-224_224-seg_augmented_additive-RGB.csv",
        "dataset/treviso-market-224_224-seg_augmented_additive_standard_08-RGB.csv"]

FILE = data[3]


def main():
    df = pd.read_csv(FILE)

    n_rows: int = ceil(sqrt(len(df)))
    n_cols: int = floor(sqrt(len(df)))

    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols)
    df = df.append([pd.DataFrame([[1.0, 1.0, 1.0, 0.0]],
                                 columns=["r", "g", "b", "y"])] * ((n_cols*n_rows) - len(df)),
                   ignore_index=True)
    for (index, image), ax in zip(df.iterrows(), axs.flatten()):
        # color = yuv2rgb(image.to_numpy())
        color = image.to_numpy()[:3]
        color_block = np.array([[color] * 25] * 25)
        ax.axis('off')
        ax.imshow(color_block)
    plt.axis('off')
    plt.show(bbox_inches='tight')


if __name__ == "__main__":
    main()
