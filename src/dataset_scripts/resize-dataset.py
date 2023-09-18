import os
from pathlib import Path
from typing import List, Dict

import yaml
from PIL import Image
from torchvision.transforms import transforms
from torchvision.transforms.functional import InterpolationMode
from tqdm import tqdm


def main(folder: str = "treviso-market"):
    # ------------------
    with open("params/images.yml", 'r') as f:
        image_parameters: Dict = yaml.load(f, Loader=yaml.SafeLoader)
    width, height = image_parameters["size"].values()
    # ------------------
    dataset_path: Path = Path("dataset")
    raw_path: Path = dataset_path / folder
    sub_folders: List[str] = os.listdir(raw_path)
    # ------------------
    resize_transform = transforms.Resize((width, height), interpolation=InterpolationMode.BILINEAR)
    resized_path: Path = dataset_path / f"{folder}-{width}_{height}"
    resized_path.mkdir(exist_ok=True)
    # ------------------
    for folder in sub_folders:
        if folder == ".DS_Store":
            continue
        images: List[str] = os.listdir(raw_path / folder)
        resized_subfolder_path: Path = resized_path / folder
        resized_subfolder_path.mkdir(exist_ok=True)
        for img_path in tqdm(images, f"Processing class {folder}..."):
            image = Image.open(raw_path / folder / img_path).convert('RGB')
            resized = resize_transform(image)
            resized.save(resized_subfolder_path / img_path)


if __name__ == "__main__":
    # main("treviso-market")
    main("fruit-standardized/Training")
    main("fruit-standardized/Test")
