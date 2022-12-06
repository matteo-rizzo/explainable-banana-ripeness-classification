import os
import shutil
from pathlib import Path
from typing import List

from tqdm import tqdm


def main(split: str = "Training"):
    path_to_fruit: Path = Path("dataset") / "fruits-360" / split
    label_list: List[str] = os.listdir(path_to_fruit)
    standardized_label_list: List[str] = [label.replace(" ", "_") for label in label_list]

    path_to_new_fruit: Path = Path("dataset") / "fruit-standardized" / split
    path_to_new_fruit.mkdir(exist_ok=True, parents=True)

    # Each label
    for label, standard_label in tqdm(zip(label_list, standardized_label_list), desc="Moving images..."):
        path_to_class: Path = path_to_fruit / label
        path_to_standard_class: Path = path_to_new_fruit / standard_label
        path_to_standard_class.mkdir(exist_ok=True, parents=True)

        examples: List[str] = os.listdir(path_to_class)
        # Each image
        for ex in examples:
            # # Images have a "100" postfix that signifies size is 100x100
            image_path = path_to_class / ex
            image_destination = path_to_standard_class / ex
            shutil.move(image_path, image_destination)


if __name__ == "__main__":
    main("Test")
