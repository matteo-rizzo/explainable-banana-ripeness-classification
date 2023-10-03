import os
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image
from torchvision.transforms import transforms as T
from tqdm import tqdm

from src.classifiers.deep_learning.classes.utils.Params import Params
from src.classifiers.deep_learning.functional.yaml_manager import load_yaml

# Select dataset
DATASET = ["treviso-market-224_224-seg", "treviso-market-224_224"][0]
MODE = ["additive", "substitutive"][1]


def apply_with_p(transformation, parameters: Dict) -> T.RandomApply:
    *params, (_, p) = parameters.items()
    return T.RandomApply([transformation(**dict(params))], p=p)


def get_transform(params: Dict, img_size: Tuple):
    return T.Compose([
        T.ToTensor(),
        # Rotates an image with random angle
        apply_with_p(T.RandomRotation, params["rotation"]),
        # Performs random affine transform on an image
        apply_with_p(T.RandomAffine, params["random_affine"]),
        # Randomly transforms the morphology of objects in images and produces a see-through-water-like effect
        apply_with_p(T.ElasticTransform, params["elastic_transform"]),
        # Crops an image at a random location
        T.Compose([apply_with_p(T.RandomCrop, params["random_crop"]),
                   T.Resize(img_size)]),
        # Randomly changes the brightness, saturation, and other properties of an image
        # apply_with_p(T.ColorJitter, params["color_jitter"]),
        # Performs gaussian blur transform on an image
        apply_with_p(T.GaussianBlur, params["gaussian_blur"]),
        # Randomly selects a rectangle region in a torch Tensor image and erases its pixels (already has p)
        T.RandomErasing(**params["random_erasing"]),
        # Performs random perspective transform on an image
        apply_with_p(T.RandomPerspective, params["random_perspective"]),
    ])


def main():
    path_to_data = os.path.join("dataset", DATASET)
    path_to_destination = Path(os.path.join("dataset", f"{DATASET}_augmented_{MODE}"))
    path_to_destination.mkdir(exist_ok=True, parents=True)

    img_details = Params.load_dataset_params(DATASET)["img_details"]
    params = load_yaml("params/augmentations/preset_1.yml")
    transform = get_transform(params, (img_details["size"]["width"],
                                       img_details["size"]["height"]))

    pillower = T.ToPILImage()
    tensower = T.ToTensor()
    for folder in os.listdir(path_to_data):
        if folder == ".DS_Store":
            continue
        path_to_class = os.path.join(path_to_data, folder)
        for file_name in tqdm(os.listdir(path_to_class), desc=f" Transforming class "):
            image = Image.open(os.path.join(path_to_class, file_name)).convert('RGB')
            label = folder
            path_to_save = path_to_destination / label
            path_to_save.mkdir(exist_ok=True, parents=True)
            transformed_image = transform(image)
            # If transformation has been applied
            if MODE == "additive":
                if not (transformed_image == tensower(image)).all():
                    transformed_image = pillower(transformed_image)
                    transformed_image.save(path_to_save / f"augmented_{file_name}")
                image.save(path_to_save / file_name)
            elif MODE == "substitutive":
                transformed_image = pillower(transformed_image)
                transformed_image.save(path_to_save / file_name)
            else:
                raise ValueError


if __name__ == "__main__":
    main()
