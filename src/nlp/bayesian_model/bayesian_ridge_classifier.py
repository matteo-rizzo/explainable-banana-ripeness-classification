import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.linear_model import BayesianRidge, RidgeClassifier


class BayesianRidgeClassifier(BayesianRidge, ClassifierMixin):
    def __init__(self, threshold=0.5,
                 max_iter=None,  # TODO(1.5): Set to 300
                 tol=1.0e-3,
                 alpha_1=1.0e-6,
                 alpha_2=1.0e-6,
                 lambda_1=1.0e-6,
                 lambda_2=1.0e-6,
                 alpha_init=None,
                 lambda_init=None,
                 compute_score=False,
                 fit_intercept=True,
                 copy_X=True,
                 verbose=False,
                 ):
        super().__init__(max_iter=max_iter, tol=tol, alpha_1=alpha_1, alpha_2=alpha_2,
                         lambda_1=lambda_1, lambda_2=lambda_2, alpha_init=alpha_init,
                         lambda_init=lambda_init, compute_score=compute_score,
                         fit_intercept=fit_intercept, copy_X=copy_X, verbose=verbose)
        self.threshold = threshold

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y)
        return self

    def fit_with_prior(self, X, y, sample_weight=None, prior=None):
        """Fit a regularized model with a nonzero prior"""
        # https://stats.stackexchange.com/questions/24889/bayesion-priors-in-ridge-regression-with-scikit-learns-linear-model
        assert prior is not None, "you need to specify a prior"
        # Scale based on prior
        new_y = y - np.sum(prior * X, axis=1)
        super().fit(X, new_y, sample_weight=sample_weight)
        # Shift back to original scale
        super().coef_ += prior  # modifying underlying model's coefficients

    def predict(self, X, return_std=False):
        y_pred = super().predict(X, return_std)
        return (y_pred > self.threshold).astype(int)


class RidgePriorClassifier(ClassifierMixin):
    def __init__(self, **kwargs):
        self.model = RidgeClassifier(**kwargs)

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight)
        return self

    def fit_with_prior(self, X, y, sample_weight=None, prior=None):
        """Fit a regularized model with a nonzero prior"""
        # https://stats.stackexchange.com/questions/24889/bayesion-priors-in-ridge-regression-with-scikit-learns-linear-model
        assert prior is not None, "you need to specify a prior"
        # Scale based on prior
        new_y = y - np.sum(prior * X, axis=1)
        self.model.fit(X, new_y, sample_weight=sample_weight)
        # Shift back to original scale
        self.model.coef_ += prior  # modifying underlying model's coefficients
        return self

    def predict(self, X):
        return self.model.predict(X)
