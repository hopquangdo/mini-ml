import numpy as np
from models.base import BaseSupervisedModel

class BaseLinearModel(BaseSupervisedModel):
    def __init__(self):
        self.weights = None

    def _add_bias(self, X):
        return np.c_[np.ones((X.shape[0], 1)), X]

    def predict(self, X):
        X_bias = self._add_bias(X)
        return X_bias @ self.weights
