import os
import sys
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from skimage.color import rgb2yuv
from tqdm import tqdm

from classes.data.managers import BananaDataManager
from classes.utils.Params import Params

# Select color space and dataset
MODE = ["YUV", "RGB"][1]
DATASET = ["treviso-market-224_224-seg", "treviso-market-224_224",
           "treviso-market-224_224-seg_augmented_additive",
           "treviso-market-224_224-seg_augmented_additive_standard"][3]


def rgb_mean(x: np.ndarray, data: Dict[str, List[float]]):
    # TODO: check if all this nanmean are correct
    # Put channels to nan where they are all masked
    x[~x.any(axis=-1)] = np.nan
    # Take mean, resulting in mean over 3 channels for each image
    avg_color = np.nanmean(x, axis=(0, 1))
    avg_color[np.isnan(avg_color)] = .0

    if not data:
        data = {"r": [], "g": [], "b": [], "y": []}

    data["r"].append(avg_color[0])
    data["g"].append(avg_color[1])
    data["b"].append(avg_color[2])

    return data


def uv_mean(x: np.ndarray, data: Dict[str, List[float]]):
    x = rgb2yuv(x, channel_axis=-1)

    # plt.imshow(x[:, :, 2])
    # plt.show()

    x[~x.any(axis=-1)] = np.nan
    avg_color = np.nanmean(x, axis=(0, 1))
    avg_color[np.isnan(avg_color)] = .0

    if not data:
        data = {"u": [], "v": [], "y": []}
        # data["l"] = []

    # data["l"].append(avg_color[0])
    data["u"].append(avg_color[1])
    data["v"].append(avg_color[2])

    return data


def color_features(path_to_destination: str, path_to_data: str, mode="YUV"):
    img_details = Params.load_dataset_params(DATASET)
    dataloader = BananaDataManager.get_full_dataloader(path_to_data, img_details, file_names_ok=True)

    tqdm_bar = tqdm(dataloader, total=len(dataloader), unit="batch", file=sys.stdout)
    tqdm_bar.set_description_str(" Average color  ")

    data = None
    for x, y, fn in dataloader:
        tqdm_bar.update(1)
        x = x.permute(0, 3, 2, 1).squeeze().numpy()

        if mode == "YUV":
            data = uv_mean(x, data)
        else:
            data = rgb_mean(x, data)
        data["y"].append(y.item())

    df_data = pd.DataFrame(data)
    df_data.to_csv(os.path.join(path_to_destination), index=False)
    tqdm_bar.close()
    pt = pd.concat([df_data.min()[:-1], df_data.max()[:-1]], axis=1)
    pt.columns = ["min", "max"]
    print(pt)


if __name__ == "__main__":
    path_to_data = os.path.join("dataset", DATASET)
    path_to_destination = os.path.join("dataset", f"{DATASET}-{MODE}.csv")

    print(f"\n\t Getting average colors of images at {path_to_data}")
    print(f"\t -> Saving dataset at {path_to_destination}")

    color_features(path_to_destination, path_to_data, mode=MODE)
