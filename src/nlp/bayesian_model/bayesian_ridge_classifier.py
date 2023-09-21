from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import BayesianRidge


class BayesianRidgeClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.5, **kwargs):
        self.threshold = threshold
        self.model = BayesianRidge(**kwargs)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        y_pred = self.model.predict(X)
        return (y_pred > self.threshold).astype(int)
