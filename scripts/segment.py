import os
import sys
import time

from matplotlib import pyplot as plt
from tqdm import tqdm

from classes.data.managers.DataManager import DataManager
from classes.segmentation.ModelSLIC import ModelSLIC


def main(path_to_destination: str, path_to_data: str):
    dataloader = DataManager.get_full_dataloader(path_to_data)
    segmenter = ModelSLIC()

    tqdm_bar = tqdm(dataloader, total=len(dataloader), unit="batch", file=sys.stdout)
    tqdm_bar.set_description_str(" Segmenting  ")

    masked_images = []
    for x, y in dataloader:
        x = x.permute(0, 3, 2, 1).numpy()
        tqdm_bar.update(1)
        masked_image = segmenter.predict(x)
        masked_images.append(masked_image)
        plt.imshow(masked_image.squeeze() * 255)
        plt.show()
        path_to_label = os.path.join(path_to_destination, str(y.numpy().data))
        os.makedirs(path_to_label, exist_ok=True)
        plt.savefig(os.path.join(path_to_label))

    tqdm_bar.close()


if __name__ == "__main__":
    path_to_data = os.path.join("dataset", "treviso-market-224_224")
    path_to_destination = os.path.join("dataset", f"treviso-market-224_224-seg_{time.time}")
    os.makedirs(path_to_destination, exist_ok=True)

    print(f"\n\t Segmenting images at {path_to_data}")
    print(f"\t -> Saving segmented images at {path_to_destination}")

    main(path_to_destination, path_to_data)
