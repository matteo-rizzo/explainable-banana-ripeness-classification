import os
import sys
import time

from matplotlib import pyplot as plt
from tqdm import tqdm
from skimage.transform import resize
from classes.data.managers import BananaDataManager
from classes.segmentation.ModelSLIC import ModelSLIC


def main(path_to_destination: str, path_to_data: str):
    dataloader = BananaDataManager.get_full_dataloader(path_to_data, file_names_ok=True)
    segmenter = ModelSLIC()

    tqdm_bar = tqdm(dataloader, total=len(dataloader), unit="batch", file=sys.stdout)
    tqdm_bar.set_description_str(" Segmenting  ")

    for x, y, fn in dataloader:
        x = x.permute(0, 3, 2, 1).numpy()
        tqdm_bar.update(1)
        masked_image = segmenter.predict(x).squeeze()
        masked_image = resize(masked_image, (224, 224, 3), preserve_range=True)
        plt.imshow(masked_image)
        path_to_label = os.path.join(path_to_destination, str(y.item()))
        os.makedirs(path_to_label, exist_ok=True)
        plt.savefig(os.path.join(path_to_label, fn[0]))

    tqdm_bar.close()


if __name__ == "__main__":
    path_to_data = os.path.join("dataset", "treviso-market")
    path_to_destination = os.path.join("dataset", f"treviso-market-224_224-seg_{time.time()}")
    os.makedirs(path_to_destination, exist_ok=True)

    print(f"\n\t Segmenting images at {path_to_data}")
    print(f"\t -> Saving segmented images at {path_to_destination}")

    main(path_to_destination, path_to_data)
