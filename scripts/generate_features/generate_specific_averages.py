from pathlib import Path
import torch
import pandas as pd
from PIL import Image
from torchvision.transforms import transforms as T
from scripts.generate_features.average_color import rgb_mean
from utils.Params import Params

DATASET = "treviso-market-224_224-seg_augmented_additive_standard_08"
IMG_ID = 97


def main():
    img_details = Params.load_dataset_params(DATASET)

    path_to_dataset: Path = Path(img_details["paths"]["dataset_dir"])

    data = None
    tensorize = T.ToTensor()
    for i in range(4):
        img = path_to_dataset / f"{i}" / f"{IMG_ID}.png"
        x = tensorize(Image.open(img).convert('RGB'))
        x = (torch.unsqueeze(x, 0)).permute(0, 3, 2, 1).squeeze().numpy()
        data = rgb_mean(x, data)
        data["y"].append(i)
    df_data = pd.DataFrame(data)

    df_data = pd.DataFrame(data)
    df_data.to_csv(f"plots/{IMG_ID}_averages.csv", index=False)



if __name__ == "__main__":
    main()
