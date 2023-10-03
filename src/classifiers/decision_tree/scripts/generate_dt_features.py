import os
import sys

import pandas as pd
from tqdm import tqdm

from src.classifiers.decision_tree.scripts.banana_features import rgb_mean, uv_mean, brownness
from src.classifiers.deep_learning.classes.data.managers import BananaDataManager
from src.classifiers.deep_learning.classes.utils.Params import Params

# Select color space and dataset
MODE = ["YUV", "RGB"][0]
DATASET = ["treviso-market-224_224-seg",  # 0
           "treviso-market-224_224",  # 1
           "treviso-market-224_224-seg_augmented_additive",  # 2
           "treviso-market-224_224-seg_augmented_additive_standard_05",  # 3
           "treviso-market-224_224-seg_augmented_additive_standard_065",  # 4
           "treviso-market-224_224-seg_augmented_additive_standard_08"][2]  # 5


def color_features(path_to_destination: str, path_to_data: str, mode="YUV", add_brownness: bool = True):
    img_details = Params.load_dataset_params(DATASET)
    dataloader = BananaDataManager.get_full_dataloader(path_to_data, img_details, file_names_ok=True)

    tqdm_bar = tqdm(dataloader, total=len(dataloader), unit="batch", file=sys.stdout)
    tqdm_bar.set_description_str(" Average color  ")

    data = {"u": [], "v": []} if mode == "YUV" else {"r": [], "g": [], "b": []}
    if add_brownness:
        data.update(**{"bu": [], "bv": [], "bn": []})
    data.update(**{"y": []})
    for x, y, fn in dataloader:
        tqdm_bar.update(1)
        x = x.permute(0, 3, 2, 1).squeeze().numpy()

        if mode == "YUV":
            data = uv_mean(x, data)
        else:
            data = rgb_mean(x, data)
        data = brownness(x, data, strategy="contrast")
        data["y"].append(y.item())

    data = {k: v for k, v in data.items() if v}
    df_data = pd.DataFrame(data)
    df_data.to_csv(os.path.join(path_to_destination), index=False)
    tqdm_bar.close()
    pt = pd.concat([df_data.min(), df_data.max()], axis=1)
    pt.columns = ["min", "max"]
    print(pt)


if __name__ == "__main__":
    path_to_data = os.path.join("dataset", DATASET)
    path_to_destination = os.path.join("dataset", f"{DATASET}-{MODE}.csv")

    print(f"\n\t Getting average colors of images at {path_to_data}")
    print(f"\t -> Saving dataset at {path_to_destination}")

    color_features(path_to_destination, path_to_data, mode=MODE)
