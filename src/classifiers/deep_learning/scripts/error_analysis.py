import sys

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tqdm import tqdm

from src.classifiers.deep_learning.classes.core.Model import Model
from src.classifiers.deep_learning.classes.data.managers import BananaDataManager
from src.classifiers.deep_learning.classes.factories.ModelFactory import ModelFactory
from src.classifiers.deep_learning.classes.utils.Params import Params


def analyze_errors(model: Model, data_params, train_params, device: torch.device):
    manager = BananaDataManager(data_params)
    manager.generate_split()
    dataloader = manager.load_split(0)['test']

    tqdm_bar = tqdm(dataloader, total=len(dataloader), unit="batch", file=sys.stdout)
    tqdm_bar.set_description_str(" Evaluating  ")

    wrong_predictions = []
    total_true = []
    total_pred = []
    with torch.no_grad():
        for x, y_true in dataloader:
            tqdm_bar.update(1)
            y_pred = torch.softmax(model.predict(x).to(device), dim=1)
            total_pred.append(torch.argmax(y_pred.cpu().detach(), dim=1))
            total_true.append(y_true.cpu().detach())
            true = list(y_true.numpy())
            pred = list(np.argmax(y_pred.cpu().detach(), axis=1))
            for t, p in zip(true, pred):
                if t != p:
                    wrong_predictions.append({"true": t, "pred": p.item()})

        tqdm_bar.close()

    y_true = torch.stack(total_true, dim=0).flatten()
    y_pred = torch.stack(total_pred, dim=0).flatten()

    classes = ["0", "1", "2", "3"]
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)

    disp.plot()

    plt.show()

    df = pd.DataFrame(wrong_predictions)
    print("----------------------------------------------------------------")
    print(" Error Analysis Overview: ")
    print(df)
    print(f"\t - Number of errors: {len(df)} (on {len(dataloader) * train_params['batch_size']} samples)")
    print("----------------------------------------------------------------")


def main():
    # --- Parameters ---
    device_type = "cuda:0"
    model_type = "cnn"
    train_params, data_params, _, _ = Params.load()
    model_params = Params.load_network_params(model_type)
    device = torch.device(device_type)
    # --- Load model ---
    model_params["device"] = device
    model = ModelFactory().get(model_type, model_params)
    model.load('results/treviso-market-224_224-seg_augmented_additive_cnn_Tue_Dec_20_11-46-15_2022/seed_3/models/cnn_fold_0.pth')
    model.evaluation_mode()
    # --- Proceed to inference ---
    print(f"\t Computing inference time for model {model_type} on {device_type}")
    print(f"\t -> Using data at {data_params['dataset']['paths']['dataset_dir']}")

    analyze_errors(model, data_params, train_params, device)


if __name__ == "__main__":
    main()
