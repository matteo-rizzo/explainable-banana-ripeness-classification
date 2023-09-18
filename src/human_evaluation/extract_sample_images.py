"""

Extract sample of images from dataset

"""
import os
import random
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

dataset_name: str = "treviso-market-224_224"
K: int = 75


def extract() -> None:
    data_folder = Path("dataset") / dataset_name
    target_folder = Path("dataset") / f"sample"

    data = dict()

    for subdir, dirs, files in os.walk(data_folder):
        class_folder = os.path.split(subdir.strip("/"))[1]
        try:
            class_folder = int(class_folder)
        except ValueError:
            class_folder = None

        for file in tqdm(files):
            tmp = os.path.splitext(file)
            if len(tmp) == 2 and tmp[1] == ".png" and class_folder is not None:
                full_name = os.path.join(subdir, file)
                if class_folder not in data:
                    data[class_folder] = list()
                data[class_folder].append(full_name)

    # Shuffle set of per-class images

    for k, images in data.items():
        random.shuffle(images)

    # Sample images and create gt vector

    flat_list = list()
    gt = list()
    for k, images in data.items():
        sample = images[:K]
        flat_list.extend(sample)
        gt.extend([k] * len(sample))

    assert len(gt) == len(flat_list), "Error no match"

    # Shuffle flat list of final sample

    all_data = list(zip(flat_list, gt))
    random.shuffle(all_data)
    random.shuffle(all_data)
    flat_list, gt = zip(*all_data)

    # Rename and write all images

    target_folder.mkdir(parents=True, exist_ok=True)
    i: int = 0
    for img_path in flat_list:
        new_name = target_folder / f"{i}.png"
        shutil.copy(img_path, new_name)
        i += 1

    # Write gt to file

    assert len(gt) == len(flat_list) == K * 4, "Error: wrong number of images"

    new_name = target_folder / "truth.csv"
    pd.Series(gt).to_csv(new_name, index=False, header=False, sep=",")


if __name__ == "__main__":
    extract()
