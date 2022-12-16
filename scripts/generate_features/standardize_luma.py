import os
from pathlib import Path

import numpy as np
from skimage.color import rgb2yuv, yuv2rgb
from skimage.io import imread, imsave

DATASET = ["treviso-market-224_224-seg", "treviso-market-224_224",
           "treviso-market-224_224-seg_augmented_additive", "treviso-market-224_224-seg_augmented_substitutive"][0]

OUT_NAME = "standard"


def standardize():
    data_folder = Path("dataset") / DATASET
    target_folder = Path("dataset") / f"{DATASET}_{OUT_NAME}"
    for subdir, dirs, files in os.walk(data_folder):
        for file in files:
            tmp = os.path.splitext(file)
            if len(tmp) == 2 and tmp[1] == ".png":
                img = imread(os.path.join(subdir, file), as_gray=False)[:, :, :-1] / 255  # (224, 224, 3)
                assert 0 <= img.max() <= 1.0
                img = rgb2yuv(img)
                img[:, :, 0] = 0.5
                img = yuv2rgb(img)

                img = (img * 255).astype(np.uint8)
                out_path = target_folder / subdir
                out_path = target_folder / os.path.basename(os.path.normpath(out_path))
                out_path.mkdir(parents=True, exist_ok=True)
                out_path /= file
                imsave(out_path, img, check_contrast=False)


if __name__ == "__main__":
    standardize()
