import os
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

from classes.core.Model import Model
from classes.data.managers import BananaDataManager
from classes.factories.ModelFactory import ModelFactory
from classes.utils.Params import Params


def main(model: Model, path_to_data: str, device: torch.device):
    dataloader = BananaDataManager.get_full_dataloader(path_to_data)

    tqdm_bar = tqdm(dataloader, total=len(dataloader), unit="batch", file=sys.stdout)
    tqdm_bar.set_description_str(" Evaluating  ")

    execution_times = []

    with torch.no_grad():
        for x, _ in dataloader:
            tqdm_bar.update(1)
            start_time = time.perf_counter()
            _ = torch.softmax(model.predict(x).to(device), dim=1)
            end_time = time.perf_counter()
            execution_times.append(end_time - start_time)

        tqdm_bar.close()

    print("----------------------------------------------------------------")
    print("\n Overview: ")
    print(f"\t - Number of items .......... : {len(dataloader)}")
    print(f"\t - Avg execution_time........ : {(np.mean(execution_times)):.4f}ms")
    print(f"\t - Std Dev execution time ... : {(np.std(execution_times)):.4f}ms")
    print("----------------------------------------------------------------")


if __name__ == "__main__":
    device_type = "cpu"
    device = torch.device(device_type)

    model_type = "mobilenet_v2"
    model_params = Params.load_network_params(model_type)
    model_params["device"] = device
    model = ModelFactory().get(model_type, model_params)
    model.load(os.path.join("trained_models", "mobilenet_v2.pth"))
    model.evaluation_mode()

    path_to_data = os.path.join("dataset", "treviso-market", "preprocessed")

    print(f"\n\t Computing inference time for model {model_type} on {device_type}")
    print(f"\t -> Using data at {path_to_data}")

    main(model, path_to_data, device)
