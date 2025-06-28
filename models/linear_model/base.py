import numpy as np
from models.core import BaseSupervisedModel

class BaseLinearModel(BaseSupervisedModel):
    def __init__(self):
        self.weights = None

    def _add_bias(self, X):
        n_samples = X.shape[0]
        bias_column = np.ones((n_samples, 1))
        return np.concatenate((bias_column, X), axis=1)
