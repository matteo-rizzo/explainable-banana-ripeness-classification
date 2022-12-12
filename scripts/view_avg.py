import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    df = pd.read_csv("dataset/treviso-market-224_224-hull-seg-RGB.csv")
    fig, axs = plt.subplots(nrows=31, ncols=30)

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

