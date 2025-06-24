import numpy as np
from models.neighbors.base import BaseKNN


class KNeighborsRegressor(BaseKNN):
    """
    KNN cho hồi quy.
    """

    def _predict_one(self, x):
        nearest = self._get_k_nearest_indices(x)
        return np.mean(self.y_train[nearest])
