import os
import sys
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from classes.data.managers import BananaDataManager
from classes.utils.Params import Params


def main(path_to_destination: str, path_to_data: str):
    img_details = Params.load_dataset_params("treviso-market")["img_details"]
    dataloader = BananaDataManager.get_full_dataloader(path_to_data, img_details, file_names_ok=True)

    tqdm_bar = tqdm(dataloader, total=len(dataloader), unit="batch", file=sys.stdout)
    tqdm_bar.set_description_str(" Average color  ")

    data = {"r": [], "g": [], "b": [], "y": []}
    for x, y, fn in dataloader:
        tqdm_bar.update(1)
        x = x.permute(0, 3, 2, 1).squeeze().numpy()
        avg_color_row = np.average(x, axis=0)
        avg_color = np.average(avg_color_row, axis=0)
        data["r"].append(avg_color[0])
        data["g"].append(avg_color[1])
        data["b"].append(avg_color[2])
        data["y"].append(y.item())

    pd.DataFrame(data).to_csv(os.path.join(path_to_destination), index=False)
    tqdm_bar.close()


if __name__ == "__main__":
    path_to_data = os.path.join("dataset", "treviso-market", "preprocessed")
    path_to_destination = os.path.join("dataset", f"treviso-market-224_224-avg_col-{time.time()}.csv")

    print(f"\n\t Getting average colors of images at {path_to_data}")
    print(f"\t -> Saving dataset at {path_to_destination}")

    main(path_to_destination, path_to_data)
