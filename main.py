import os
import random
import re
import time

import numpy as np
import pandas as pd
import torch

from classes.core.CrossValidator import CrossValidator
from classes.data.DataManager import DataManager
from classes.utils.Params import Params


def set_random_seed(seed: int, device: torch.device):
    torch.manual_seed(seed)
    if device.type == 'cuda:3':
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device(device_type: str) -> torch.device:
    """
    Returns the device specified in the experiments parameters (if available, else fallback to a "cpu" device)
    :param device_type: the id of the selected device (if cuda device, must match the regex "cuda:\d"
    :return: the device specified in the experiments parameters (if available, else fallback to a "cpu" device)
    """
    if device_type == "cpu":
        return torch.device("cpu")

    if re.match(r"\bcuda:\b\d+", device_type):
        if not torch.cuda.is_available():
            print(f"WARNING: running on cpu since device {device_type} is not available")
            return torch.device("cpu")
        return torch.device(device_type)

    raise ValueError(f"ERROR: {device_type} is not a valid device! Supported device are 'cpu' and 'cuda:n'")


def main():
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    train_params, data_params, num_seeds, device_type = Params.load()

    device = get_device(device_type)
    set_random_seed(0, device)
    train_params["device"] = device

    network_type = train_params["network_type"]
    dataset_name = data_params["dataset"]["name"]

    print(f"\n\n==========================================================\n"
          f"            Experiment on {dataset_name} using {network_type}                       \n"
          f"==========================================================\n\n")

    print(f"\t Using Torch version ... : {torch.__version__}")
    print(f"\t Running on device ..... : {device}\n")

    experiment_id = f"{data_params['dataset']['name']}_{network_type}_{str(time.time())}"
    path_to_results = os.path.join("results", experiment_id)
    os.makedirs(path_to_results)
    Params.save_experiment_params(path_to_results, network_type, dataset_name)

    test_scores = []
    start_time = time.time()

    for seed in range(num_seeds):
        print(f"\n\n==========================================================\n"
              f"                      Seed {seed + 1} / {num_seeds}                       \n"
              f"==========================================================\n")

        data_manager = DataManager(data_params, network_type)
        use_cv_metadata = data_params["cv"]["use_cv_metadata"]

        if use_cv_metadata:
            path_to_metadata = data_params["dataset"]["paths"]["cv_metadata"]
            data_manager.reload_split(path_to_metadata, seed + 1)
        else:
            data_manager.generate_split()
            data_manager.save_split_to_file(path_to_results, seed + 1)

        data_manager.print_split_info()

        cv = CrossValidator(data_manager, path_to_results, train_params)

        set_random_seed(seed, device)
        test_scores += [cv.validate(seed + 1)]

        print("\n................................................................\n"
              "                        Finished CV                  \n"
              "................................................................\n")

    print(f"\n Average test results for {num_seeds} seeds \n")
    avg_seeds_test = pd.DataFrame(test_scores)
    avg_seeds_test.insert(0, "seed", list(range(1, num_seeds + 1)))
    avg_seeds_test.to_csv(os.path.join(path_to_results, "avg_seeds_test.csv"), index=False)
    print(avg_seeds_test)
    elapsed: float = (time.time() - start_time) / 60
    print("\n\n==========================================================\n"
          f"            Finished experiment in {elapsed:.2f}m              \n"
          "==========================================================\n")


if __name__ == "__main__":
    main()
