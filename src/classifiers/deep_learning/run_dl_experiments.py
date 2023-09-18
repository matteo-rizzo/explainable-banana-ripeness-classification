import os
import time
from pathlib import Path

import pandas as pd
import torch

from classes.core.CrossValidator import CrossValidator
from classes.factories.DataManagerFactory import DataManagerFactory
from classes.utils.Params import Params
from functional.setup import set_random_seed, get_device


def main():
    """    Main entry point for scripts    """
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load all parameters from file
    train_params, data_params, num_seeds, device_type = Params.load()

    # Setup devices and seeds for training
    device = get_device(device_type)
    set_random_seed(0, device)
    train_params["device"] = device

    network_type = train_params["network_type"]
    dataset_name = data_params["dataset"]["name"]

    print(
        f"\n======================================================{'=' * len(network_type) + '=' * len(dataset_name)}\n"
        f"            Experiment on {dataset_name} using {network_type}                       \n"
        f"======================================================{'=' * len(network_type) + '=' * len(dataset_name)}")

    print(f"\t Using Torch version ... : {torch.__version__}")
    print(f"\t Running on device ..... : {device}")

    # Initialize folder for result output
    experiment_id = f"{data_params['dataset']['name']}" \
                    f"_{network_type}_{time.ctime().replace(' ', '_').replace(':', '-')}"
    path_to_results = Path("../../results") / experiment_id
    path_to_results.mkdir(parents=True)
    Params.save_experiment_params(path_to_results, network_type, dataset_name)

    test_scores = []
    start_time: float = time.perf_counter()

    # Loop through seeds
    for seed in range(num_seeds):
        print(f"\n==============================================================\n"
              f"                      Seed {seed + 1} / {num_seeds}                       \n"
              f"==============================================================")
        data_manager = DataManagerFactory().get(data_params)
        # BananaDataManager(data_params)
        use_cv_metadata: bool = data_params["cv"]["use_cv_metadata"]

        if use_cv_metadata:
            path_to_metadata = data_params["dataset"]["paths"]["cv_metadata"]
            data_manager.reload_split(path_to_metadata, seed + 1)
        else:
            data_manager.generate_split()
            data_manager.save_split_to_file(path_to_results, seed + 1)

        data_manager.print_split_info()

        # Initialize the cross validator, which will also train the models
        cv = CrossValidator(data_manager, path_to_results, train_params)

        set_random_seed(seed, device)
        test_scores += [cv.validate(seed + 1)]

    print("\n................................................................\n"
          "                        Finished CV                  \n"
          "................................................................")

    print(f" Average test results for {num_seeds} seeds ")
    avg_seeds_test = pd.DataFrame(test_scores)
    avg_seeds_test.insert(0, "seed", list(range(1, num_seeds + 1)))
    avg_seeds_test.to_csv(os.path.join(path_to_results, "avg_seeds_test.csv"), index=False)
    print(avg_seeds_test)
    elapsed: float = (time.perf_counter() - start_time) / 60
    print("\n\n==========================================================\n"
          f"            Finished experiment in {elapsed:.2f}m              \n"
          "==========================================================\n")


if __name__ == "__main__":
    main()
