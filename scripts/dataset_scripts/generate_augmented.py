import os
import sys
from pathlib import Path

from matplotlib import pyplot as plt
from skimage import io
from tqdm import tqdm

from classes.data.managers import BananaDataManager
from classes.utils.Params import Params

# Select dataset
DATASET = ["treviso-market-224_224-seg", "treviso-market-224_224"][0]


def main():
    path_to_data = os.path.join("dataset", DATASET)
    path_to_destination = Path(os.path.join("dataset", f"{DATASET}_augmented"))
    path_to_destination.mkdir(exist_ok=True, parents=True)

    img_details = Params.load_dataset_params(DATASET)["img_details"]
    dataloader = BananaDataManager.get_full_dataloader(path_to_data, img_details, file_names_ok=True)

    tqdm_bar = tqdm(dataloader, total=len(dataloader), unit="batch", file=sys.stdout)
    tqdm_bar.set_description_str(" Augmenting images  ")

    data = []
    for x, y, fn in dataloader:
        tqdm_bar.update(1)
        # Tensor 2 ndarray
        img = x.permute(0, 3, 2, 1).squeeze().numpy()
        io.imshow(img)
        plt.show()
        print("")


if __name__ == "__main__":
    main()
