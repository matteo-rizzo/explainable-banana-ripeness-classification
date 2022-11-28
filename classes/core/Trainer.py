from typing import Dict, Tuple

from classes.core.Evaluator import Evaluator
from classes.factories.ModelFactory import ModelFactory
from classes.utils.Params import Params


class Trainer:

    def __init__(self, train_params: Dict, path_to_best_model: str):
        """
        :param train_params: the train related params in the experiment.json file
        :param path_to_best_model: the path at which the best model is saved during train
        """
        self.__path_to_best_model = path_to_best_model

        self.__device = train_params["device"]
        self.__epochs = train_params["epochs"]
        self.__optimizer_type = train_params["optimizer"]
        self.__lr = train_params["learning_rate"]

        self.__log_every = train_params["log_every"]
        self.__evaluate_every = train_params["evaluate_every"]

        self.__patience = train_params["early_stopping"]["patience"]
        self.__es_metric = train_params["early_stopping"]["metrics"]
        self.__es_metric_trend = train_params["early_stopping"]["metrics_trend"]
        self.__es_metric_best_value = 0.0 if self.__es_metric_trend == "increasing" else 1000
        self.__epochs_no_improvement = 0

        network_type, criterion_type = train_params["network_type"], train_params["criterion"]

        network_params = Params.load_network_params(network_type)
        network_params["device"] = self.__device

        self.__model = ModelFactory().get(network_type, network_params)
        self.__model.set_optimizer(self.__optimizer_type, self.__lr)
        self.__model.set_criterion(criterion_type)

        self.__evaluator = Evaluator(self.__device)

    def train(self, data: Dict) -> Tuple:
        """
        Trains the model according to the established parameters and the given data
        :param data: a dictionary of data loaders containing train, val and test data
        :return: the evaluation metrics of the training and the trained model
        """
        print("\n Training the model...")

        self.__model.print_model_overview()

        evaluations = []
        training_loader = data["train"]

        for epoch in range(self.__epochs):
            print(f"\n *** Epoch {epoch + 1}/{self.__epochs} *** ")

            self.__model.train_mode()

            running_loss, running_accuracy = 0.0, 0.0

            for i, (x, y) in enumerate(training_loader):

                self.__model.reset_gradient()

                y = y.long().to(self.__device)
                o = self.__model.predict(x).to(self.__device)

                running_loss += self.__model.update_weights(o, y)
                running_accuracy += Evaluator.batch_accuracy(o, y)

                if not (i + 1) % self.__log_every:
                    avg_loss, avg_accuracy = running_loss / self.__log_every, running_accuracy / self.__log_every
                    print(
                        f"[ Epoch: {epoch + 1}/{self.__epochs}, batch: {i + 1} ] "
                        f"[ Loss: {avg_loss:.4f} | Accuracy: {avg_accuracy:.4f} ]")
                    running_loss, running_accuracy = 0.0, 0.0

            print(" ...........................................................")

            if not (epoch + 1) % self.__evaluate_every:
                evaluations += [self.__evaluator.evaluate(data, self.__model)]
                if self.__early_stopping_check(evaluations[-1]["metrics"]["val"][self.__es_metric]):
                    break

        print("\n Finished training!")

        return self.__model, evaluations

    def __early_stopping_check(self, metric_value: float) -> bool:
        """
        Decides whether to early stop the train based on the early stopping conditions
        @param metric_value: the monitored val metrics (e.g. auc, loss)
        @return: a flag indicating whether the training should be early stopped
        """
        if self.__es_metric_trend == "increasing":
            metrics_check = metric_value > self.__es_metric_best_value
        else:
            metrics_check = metric_value < self.__es_metric_best_value

        if metrics_check:
            print(
                f"\t Old best val {self.__es_metric}: {self.__es_metric_best_value:.4f} "
                f"| New best {self.__es_metric}: {metric_value:.4f}")

            print("\t Saving new best model...")
            self.__model.save(self.__path_to_best_model)
            print("\t -> New best model saved!")

            self.__es_metric_best_value = metric_value
            self.__epochs_no_improvement = 0
        else:
            self.__epochs_no_improvement += 1
            if self.__epochs_no_improvement == self.__patience:
                print(f" ** No decrease in val {self.__es_metric} "
                      f"for {self.__patience} evaluations. Early stopping! ** ")
                return True

        print(" Epochs without improvement: ", self.__epochs_no_improvement)
        print(" ........................................................... ")
        return False
