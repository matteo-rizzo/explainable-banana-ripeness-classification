import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from classes.core.Model import Model
from classes.data.Dataset import Dataset
from classes.data.loaders.ImageLoader import ImageLoader
from classes.factories.ModelFactory import ModelFactory
from classes.utils.Params import Params


def main(model: Model, path_to_data: str, device: torch.device):
    """
    Evaluates the saved best model against train, val and test data
    :param model: the model to be evaluated
    :param path_to_data: the path to the dataset class folders
    :return: the eval of the model on train, val and test data, including metrics, gt and preds
    """

    data_paths, labels = [], []
    for folder in os.listdir(path_to_data):
        path_to_class = os.path.join(path_to_data, folder)
        for file_name in os.listdir(path_to_class):
            data_paths.append(os.path.join(path_to_class, file_name))
            labels.append(int(folder))

    dataloader = DataLoader(Dataset(data_paths, labels, ImageLoader().load), batch_size=1)

    tqdm_bar = tqdm(dataloader, total=len(dataloader), unit="batch", file=sys.stdout)
    tqdm_bar.set_description_str(" Evaluating  ")

    execution_times = []

    with torch.no_grad():

        for i, (x, _) in enumerate(dataloader):
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
