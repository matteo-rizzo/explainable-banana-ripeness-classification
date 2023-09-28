import numpy as np
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.linear_model import BayesianRidge, RidgeClassifier, Ridge, LogisticRegression


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


class RidgePriorClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, alpha=1.0, fit_intercept=True, positive=False, copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.copy_X = copy_X
        self.max_iter = max_iter
        self.tol = tol
        self.solver = solver
        self.positive = positive
        self.random_state = random_state
        self.model = Ridge(alpha=self.alpha, fit_intercept=self.fit_intercept, positive=self.positive,
                          copy_X=self.copy_X, max_iter=self.max_iter, tol=self.tol, solver=self.solver, random_state=self.random_state)

    # def get_params(self, deep=True):
    #     return {"alpha": self.alpha, "fit_intercept": self.fit_intercept,
    #             "positive": self.positive, "copy_X": self.copy_X, "max_iter": self.max_iter, "tol": self.tol, "solver": self.solver, "random_state": self.random_state}
    #
    # def set_params(self, **parameters):
    #     for parameter, value in parameters.items():
    #         setattr(self, parameter, value)
    #     return self

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
        return (self.model.predict(X) > 0.5).astype(int)


class LogisticPriorClassifier:
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)

    def fit(self, X, y, sample_weight=None):
        self.model.fit(X, y, sample_weight)
        return self

    def fit_with_prior(self, X, y, sample_weight=None, prior=None):
        """Fit a regularized model with a nonzero prior"""
        assert prior is not None, "you need to specify a prior"
        # Scale based on prior
        new_y = y - np.sum(prior * X, axis=1)
        # Ensure new_y is binary
        new_y = (new_y > 0.5).astype(int)
        self.model.fit(X, new_y, sample_weight=sample_weight)
        # Shift back to original scale
        self.model.coef_ += prior  # modifying underlying model's coefficients
        return self

    def predict(self, X):
        return self.model.predict(X)
