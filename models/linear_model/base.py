# models/linear/base.py

import numpy as np
from models.base import BaseSupervisedModel
from models.utils import ensure_numpy


class BaseLinearModel(BaseSupervisedModel):
    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        X = ensure_numpy(X)
        y = ensure_numpy(y)

        X_bias = np.c_[np.ones((X.shape[0],)), X]
        self.weights = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y

    def _add_bias(self, X):
        return np.c_[np.ones((X.shape[0],)), X]
