import os
from pathlib import Path

import numpy as np
from skimage.color import rgb2yuv, yuv2rgb
from skimage.io import imread, imsave
from tqdm import tqdm

DATASET = ["treviso-market-224_224-seg", "treviso-market-224_224",
           "treviso-market-224_224-seg_augmented_additive"][2]

OUT_NAME = "standard_08"

STANDARD_LUMA: float = 0.8  # in (0, 1)


def standardize():
    data_folder = Path("dataset") / DATASET
    target_folder = Path("dataset") / f"{DATASET}_{OUT_NAME}"

    for subdir, dirs, files in os.walk(data_folder):
        for file in tqdm(files):
            tmp = os.path.splitext(file)
            if len(tmp) == 2 and tmp[1] == ".png":
                img = imread(os.path.join(subdir, file), as_gray=False) / 255
                if img.shape[-1] > 3:
                    img = img[:, :, :-1]  # (224, 224, 3)
                assert 0 <= img.max() <= 1.0

                # Get pixels masked by segmentation
                masked_pixels = ~img.any(axis=-1)

                # Convert to YUV
                img = rgb2yuv(img)
                # Set luma (Y) to standard value
                img[:, :, 0] = STANDARD_LUMA
                # Convert in RGB
                img = yuv2rgb(img)

                # Masked pixels should be set to 0
                img[masked_pixels] = .0

                img = (img * 255).astype(np.uint8)
                out_path = target_folder / subdir
                out_path = target_folder / os.path.basename(os.path.normpath(out_path))
                out_path.mkdir(parents=True, exist_ok=True)
                out_path /= file
                imsave(out_path, img, check_contrast=False)


if __name__ == "__main__":
    standardize()
