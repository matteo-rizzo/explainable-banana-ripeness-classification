import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from src.classifiers.deep_learning.classes.core.Model import Model
from src.classifiers.deep_learning.classes.data.managers import BananaDataManager
from src.classifiers.deep_learning.classes.factories.ModelFactory import ModelFactory
from src.classifiers.deep_learning.classes.utils.Params import Params


def infer(model: Model, data_params, path_to_data: str, device: torch.device):
    dataloader = BananaDataManager.get_full_dataloader(path_to_data, data_params)

    tqdm_bar = tqdm(dataloader, total=len(dataloader), unit="batch", file=sys.stdout)
    tqdm_bar.set_description_str(" Evaluating  ")

    execution_times = []
    with torch.no_grad():
        for x, y_true in dataloader:
            tqdm_bar.update(1)
            start_time = time.perf_counter()
            _ = torch.softmax(model.predict(x).to(device), dim=1)
            end_time = time.perf_counter()
            execution_times.append(end_time - start_time)

        tqdm_bar.close()
    print("----------------------------------------------------------------")
    print(" Inference Time Overview: ")
    print(f"\t - Number of items .......... : {len(dataloader)}")
    print(f"\t - Avg execution_time........ : {(np.mean(execution_times)):.4f}ms")
    print(f"\t - Std Dev execution time ... : {(np.std(execution_times)):.4f}ms")
    print("----------------------------------------------------------------")


def main():
    # --- Parameters ---
    device_type = "cuda:0"
    model_type = "mobilenet_v2"
    dataset = "treviso-market-224_224-seg_augmented_additive"
    data_params = Params.load_dataset_params(dataset)
    model_params = Params.load_network_params(model_type)
    device = torch.device(device_type)
    # --- Load model ---
    model_params["device"] = device
    model = ModelFactory().get(model_type, model_params)
    model.load(os.path.join("trained_models", "mobilenet_v2.pth"))
    model.evaluation_mode()
    # --- Proceed to inference ---
    path_to_data = os.path.join("dataset", dataset)

    print(f"\t Computing inference time for model {model_type} on {device_type}")
    print(f"\t -> Using data at {path_to_data}")

    infer(model, data_params, path_to_data, device)


if __name__ == "__main__":
    main()
