import numpy as np
from models.base import BaseSupervisedModel


class BaseKNN(BaseSupervisedModel):
    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        raise NotImplementedError()

    def _get_k_nearest_indices(self, x):
        if self.distance_metric == 'euclidean':
            distances = np.linalg.norm(self.X_train - x, axis=1)
        else:
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        return np.argsort(distances)[:self.k]
