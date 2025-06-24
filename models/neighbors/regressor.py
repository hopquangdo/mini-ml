import numpy as np
from models.neighbors.base import BaseKNN

class KNeighborsRegressor(BaseKNN):
    """
    KNN cho hồi quy.
    """

    def _predict_one(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        nearest = np.argsort(distances)[:self.k]
        return np.mean(self.y_train[nearest])
