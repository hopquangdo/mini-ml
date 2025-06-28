import numpy as np
from models.core import BaseSupervisedModel


class BaseSVM(BaseSupervisedModel):
    def __init__(self, C=1.0, lr=1e-3, max_iter=1000):
        self.C = C
        self.lr = lr
        self.max_iter = max_iter
        self.w = None
        self.b = 0

    def _init_weights(self, n_features):
        self.w = np.zeros(n_features)
        self.b = 0

    def _linear_kernel(self, X1, X2):
        return X1 @ X2.T

    def decision_function(self, X):
        return X @ self.w + self.b
