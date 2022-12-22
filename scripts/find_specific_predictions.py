import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

from classes.core.Model import Model
from classes.data.managers import BananaDataManager
from classes.factories.ModelFactory import ModelFactory
from classes.utils.Params import Params
from torchvision import transforms as T


def infer(model: Model, data_params, path_to_data: str, device: torch.device):
    dataloader = BananaDataManager.get_full_dataloader(path_to_data, data_params)

    tqdm_bar = tqdm(dataloader, total=len(dataloader), unit="batch", file=sys.stdout)
    tqdm_bar.set_description_str(" Evaluating  ")

    preds = []
    pillow = T.ToPILImage()
    with torch.no_grad():
        for idx, (x, y_true) in enumerate(dataloader):
            tqdm_bar.update(1)
            prediction = torch.softmax(model.predict(x).to(device), dim=1)
            preds.append(np.argmax(prediction.cpu().detach()))

        tqdm_bar.close()
    print(preds)


def main():
    # --- Parameters ---
    device_type = "cuda:0"
    model_type = "cnn"
    dataset = "cheating"
    data_params = Params.load_dataset_params(dataset)
    model_params = Params.load_network_params(model_type)
    device = torch.device(device_type)
    # --- Load model ---
    model_params["device"] = device
    model = ModelFactory().get(model_type, model_params)
    model.load(
        'results/treviso-market-224_224-seg_augmented_additive_cnn_Tue_Dec_20_11-46-15_2022/seed_3/models/cnn_fold_0.pth')
    model.evaluation_mode()
    # --- Proceed to inference ---
    path_to_data = os.path.join("dataset", dataset)

    print(f"\t Computing inference time for model {model_type} on {device_type}")
    print(f"\t -> Using data at {path_to_data}")

    infer(model, data_params, path_to_data, device)


if __name__ == "__main__":
    main()
