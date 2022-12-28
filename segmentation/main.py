import os
import sys
import time

from classes.data.managers import BananaDataManager
from classes.utils.Params import Params
from matplotlib import pyplot as plt
from skimage.transform import resize
from tqdm import tqdm

from segmentation.classes.ModelSLIC import ModelSLIC


def main(path_to_destination: str, path_to_data: str):
    img_details = Params.load_dataset_params("treviso-market")["img_details"]
    dataloader = BananaDataManager.get_full_dataloader(path_to_data, img_details, file_names_ok=True)
    segmenter = ModelSLIC()

    tqdm_bar = tqdm(dataloader, total=len(dataloader), unit="batch", file=sys.stdout)
    tqdm_bar.set_description_str(" Segmenting  ")

    for x, y, fn in dataloader:
        tqdm_bar.update(1)
        x = x.permute(0, 3, 2, 1).numpy()
        masked_image = segmenter.predict(x).squeeze()
        masked_image = resize(masked_image, (224, 224, 3), preserve_range=True)
        path_to_label = os.path.join(path_to_destination, str(y.item()))
        os.makedirs(path_to_label, exist_ok=True)
        plt.imsave(os.path.join(path_to_label, fn[0]), masked_image)

    tqdm_bar.close()


if __name__ == "__main__":
    path_to_data = os.path.join("dataset", "treviso-market", "preprocessed")
    path_to_destination = os.path.join("dataset", f"treviso-market-224_224-seg_{time.time()}")
    os.makedirs(path_to_destination, exist_ok=True)

    print(f"\n\t Segmenting images at {path_to_data}")
    print(f"\t -> Saving segmented images at {path_to_destination}")

    main(path_to_destination, path_to_data)
