import os
import time
from pathlib import Path
from typing import Union, Dict, List

import numpy as np
import torch

from src.classifiers.deep_learning.classes.core.Evaluator import Evaluator
from src.classifiers.deep_learning.classes.core.Trainer import Trainer
from src.classifiers.deep_learning.classes.data.managers import BananaDataManager
from src.classifiers.deep_learning.classes.utils.Params import Params


class CrossValidator:

    def __init__(self, data_manager: BananaDataManager, path_to_results: Union[str, Path], train_params: Dict):
        """
        :param data_manager: an instance of DataManager to load the folds from the filesystem
        :param path_to_results: the path to the directory with the results for the current experiment
        :param train_params: the params to be submitted to the Trainer instance
        """
        self.data_manager = data_manager
        self.__path_to_results = path_to_results
        self.__train_params = train_params
        self.evaluator = Evaluator(train_params["device"], num_classes=train_params["num_classes"])
        self.__paths_to_results = {}

    @staticmethod
    def __merge_metrics(metrics: List, set_type: str) -> Dict:
        """
        Averages the metrics by set type (in ["train", "val", "test"])
        :param metrics: the metrics of each processed fold
        :param set_type: set code in ["train", "val", "test"]
        :return: the input metrics averaged by set type
        """
        return {k: float(np.array([m[set_type][k] for m in metrics]).mean()) for k in metrics[0][set_type].keys()}

    def __avg_metrics(self, cv_metrics: List, save: bool = False, inplace: bool = False) -> Union[Dict, None]:
        """
        Computes the average metrics for the current CV iteration
        :param cv_metrics: the list of metrics for each processed fold of the CV iteration
        :param save: whether to save to file the average metrics
        :param inplace: whether to return the average metrics
        """
        avg_scores = {}
        for set_type in ["train", "val", "test"]:
            avg_scores[set_type] = self.__merge_metrics(cv_metrics, set_type)
            print(f" Average {set_type} metrics: ")
            for metric, value in avg_scores[set_type].items():
                print(f"\t {metric} - {''.join(['.'] * (15 - len(metric)))} : {value:.4f}")

        if save:
            Params.save(avg_scores, os.path.join(self.__paths_to_results["metrics"], "cv_average.yml"))

        if not inplace:
            return avg_scores

    def __create_paths_to_results(self, seed: int):
        """
        Creates the paths to the "metrics", "models", "preds" and "plots" folder for the current experiment.
        @param seed: the current random seed
        """
        path_to_main_dir = os.path.join(self.__path_to_results, f"seed_{str(seed)}")

        self.__paths_to_results = {
            "metrics": os.path.join(path_to_main_dir, "metrics"),
            "models": os.path.join(path_to_main_dir, "models"),
            "preds": os.path.join(path_to_main_dir, "preds"),
            "plots": os.path.join(path_to_main_dir, "plots")
        }

        for path in self.__paths_to_results.values():
            os.makedirs(path)

    def validate(self, seed: int) -> Dict:
        """
        Performs an iteration of CV for the given random seed
        :param seed: the seed number of the CV
        """
        self.__create_paths_to_results(seed)

        cv_metrics, folds_times = [], []
        zero_time = time.perf_counter()

        k = self.data_manager.get_k()

        for fold in range(k):
            print(f"\n * Processing fold {fold + 1} / {k} - seed {seed} * \n")

            model_name = self.__train_params["network_type"] + "_fold_" + str(fold)
            path_to_best_model = os.path.join(self.__paths_to_results["models"], model_name + ".pth")
            trainer = Trainer(self.__train_params, path_to_best_model)

            data = self.data_manager.load_split(fold)

            start_time = time.perf_counter()
            model, _ = trainer.train(data)
            end_time = time.perf_counter()

            best_eval = self.evaluator.evaluate(data, model, path_to_best_model)

            print(f"\n *** Finished processing fold {fold + 1} / {k}! ***\n")

            print(" Saving metrics...")
            metrics_log = f"fold_{str(fold)}.yml"
            Params.save(best_eval["metrics"], os.path.join(self.__paths_to_results["metrics"], metrics_log))
            cv_metrics.append(best_eval["metrics"])
            print("-> Metrics saved!")

            print(" Saving preds...")
            Params.save_experiment_preds(best_eval, self.__paths_to_results["preds"], fold + 1)
            print("-> Predictions saved!")

            self.__avg_metrics(cv_metrics, inplace=True)

            folds_times.append((start_time - end_time) / 60)
            estimated_time = abs(np.mean(np.array(folds_times)) * (k - fold))
            print("\n Time overview: ")
            print(f"\t - Time to train fold ............. : {(end_time - start_time) / 60:.2f}m")
            print(f"\t - Elapsed time CV time: .......... : {(end_time - zero_time) / 60:.2f}m")
            print(f"\t - Estimated time of completion ... : {estimated_time:.2f}m")
            print("----------------------------------------------------------------")

            # Necessary for sequential run. Empty cache should be automatic, but best be sure.
            del trainer, model, data
            torch.cuda.empty_cache()
        return self.__avg_metrics(cv_metrics, save=True)["test"]
