import sys
from typing import Dict

import numpy as np
import torch
import torch.utils.data
from torchmetrics.functional.classification import multiclass_f1_score, multiclass_recall, multiclass_precision
from tqdm import tqdm

from src.classifiers.deep_learning.classes.core.Model import Model


class Evaluator:
    def __init__(self, device: torch.device, num_classes: int):
        """
        :param device: the device which to run on (gpu or cpu)
        """
        self.__device = device
        self.__num_classes = num_classes

    def evaluate(self, data: Dict, model: Model, path_to_model: str = "") -> Dict:
        """
        Evaluates the saved best model against train, val and test data
        :param data: a dictionary tuple containing the data loaders for train, val and test
        :param model: the model to be evaluated
        :param path_to_model: the path to the saved serialization of the best model
        :return: the eval of the model on train, val and test data, including metrics, gt and preds
        """
        model.evaluation_mode()

        if path_to_model != "":
            model.load(path_to_model)

        metrics, gt, preds = {}, {}, {}
        for set_type, dataloader in data.items():

            # Visual progress bar
            tqdm_bar = tqdm(dataloader, total=len(dataloader), unit="batch", file=sys.stdout)
            tqdm_bar.set_description_str(" Evaluating  ")

            loss, accuracy, y_scores, y_true = [], [], [], []

            with torch.no_grad():

                for i, (x, y) in enumerate(dataloader):
                    tqdm_bar.update(1)
                    # Prepare true and logits
                    y = y.long().to(self.__device)
                    output_logits = model.predict(x).to(self.__device)
                    # Append loss and accuracy values for current batch
                    loss.append([model.get_loss(output_logits, y)])
                    accuracy.append([self.batch_accuracy(output_logits, y)])
                    # Calculate output probabilities
                    output_probabilities = torch.softmax(output_logits, dim=1)
                    # Pass scores and true to cpu
                    y_scores.append(output_probabilities.cpu())
                    y_true.append(y.cpu())
                tqdm_bar.close()
            # Find predictions
            y_pred, y_true = torch.stack([torch.argmax(y, dim=1) for y in y_scores]), torch.stack(y_true)
            # Compute macro recall / prec / f1
            set_metrics = self.__compute_metrics(y_true, y_pred)
            # Mean accuracy and loss over all batches
            set_metrics["accuracy"], set_metrics["loss"] = float(np.mean(accuracy)), float(np.mean(loss))

            print(f"{set_type.upper()} metrics:")
            for metric, value in set_metrics.items():
                print(f"\t {metric} - {''.join(['.'] * (15 - len(metric)))} : {value:.4f}")

            metrics[set_type], gt[set_type], preds[set_type] = set_metrics, y_true, y_pred

        return {"metrics": metrics, "gt": gt, "preds": preds}

    @staticmethod
    def batch_accuracy(output_logits: torch.Tensor, y: torch.Tensor) -> float:
        """
        Computes the accuracy of the preds over the items in a single batch
        :param output_logits: the logit output of datum in the batch
        :param y: the correct class index of each datum
        :return the percentage of correct preds as a value in [0,1]
        """
        output_probabilities = torch.softmax(output_logits, dim=1)
        y_pred = torch.argmax(output_probabilities, dim=1)

        return (torch.sum(y_pred == y) / y.shape[0]).item()

    def __compute_metrics(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict:
        """
        Computes the metrics for the given preds and labels
        :param y_true: the ground-truth labels
        :param y_pred: the preds of the model
        :return: the following metrics in a Dict: accuracy / macro precision / macro recall / macro F1
        """
        return {
            "precision": float(multiclass_precision(y_pred, y_true, num_classes=self.__num_classes)),
            "recall": float(multiclass_recall(y_pred, y_true, num_classes=self.__num_classes)),
            "macro_f1": float(multiclass_f1_score(y_pred, y_true, num_classes=self.__num_classes))
        }
