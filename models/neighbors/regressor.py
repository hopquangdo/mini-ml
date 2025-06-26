import numpy as np
from models.neighbors.base import BaseKNN


class KNeighborsRegressor(BaseKNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _predict_one(self, x):
        indices, distances = self._get_k_nearest(x)
        values = self.y_train[indices]

        if self.weights == 'uniform':
            return np.mean(values)

        # average = sum(value_i * weight_i) / sum(weight_i)
        return np.average(values, weights=self._compute_weights(distances, values))
