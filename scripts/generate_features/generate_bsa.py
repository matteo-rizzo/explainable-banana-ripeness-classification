import os
import sys

import pandas as pd
from skimage.color import rgb2lab
from skimage.morphology import opening
from tqdm import tqdm

from classes.data.managers import BananaDataManager
from classes.utils.Params import Params

DATASET = ["treviso-market-224_224-seg", "treviso-market-224_224",
           "treviso-market-224_224-seg_augmented_additive", "treviso-market-224_224-seg_augmented_substitutive"][3]


def rgb_to_scaled_lab(image_rgb):
    image_lab = rgb2lab(image_rgb)
    # Rescale LAB images because of skimage shenanigans.
    # This was actually the solution of one of the authors
    lab_scaled = (image_lab + [0, 128, 128]) / [100, 255, 255]
    return lab_scaled


def binarize_lab(image_lab):
    # Threshold taken by paper, works well
    threshold = 130 / 255
    l, a, b = image_lab[:, :, 0], image_lab[:, :, 1], image_lab[:, :, 2]
    # Binarize on "a" channel
    binarized_image = (a > threshold).astype(int)
    return binarized_image


def calculate_bsa(img):
    lab = rgb_to_scaled_lab(img)
    # ---- Binary mask image ----
    bin_msk = binarize_lab(lab)
    # --- Morphed mask image ----
    morphed = opening(bin_msk)
    # ---------------------------
    bsa = (morphed.sum() / morphed.size) * 100
    return bsa


def main():
    path_to_data = os.path.join("dataset", DATASET)
    path_to_destination = os.path.join("dataset", f"{DATASET}_bsa.csv")

    img_details = Params.load_dataset_params(DATASET)
    dataloader = BananaDataManager.get_full_dataloader(path_to_data, img_details, file_names_ok=True)

    tqdm_bar = tqdm(dataloader, total=len(dataloader), unit="batch", file=sys.stdout)
    tqdm_bar.set_description_str(" Finding BSA for images  ")

    data = []
    for x, y, fn in dataloader:
        tqdm_bar.update(1)
        # Tensor 2 ndarray
        img = x.permute(0, 3, 2, 1).squeeze().numpy()
        bsa = calculate_bsa(img)
        data.append(bsa)

    pd.DataFrame(data, columns=["bsa"]).to_csv(os.path.join(path_to_destination), index=False)
    tqdm_bar.close()


if __name__ == "__main__":
    main()
