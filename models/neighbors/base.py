import numpy as np
from models.core import BaseSupervisedModel


class BaseKNN(BaseSupervisedModel):
    def __init__(self, k=3, distance_metric='euclidean', weights="uniform"):
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        raise NotImplementedError()

    def _compute_distances(self, x):
        if self.distance_metric == 'euclidean':
            return np.linalg.norm(self.X_train - x, axis=1)
        return np.sum(np.abs(self.X_train - x), axis=1)

    def _compute_weights(self, distances, labels):
        distances = np.where(distances == 0, 1e-8, distances)
        return 1.0 / distances

    def _get_k_nearest(self, x):
        distances = self._compute_distances(x)
        indices = np.argsort(distances)[:self.k]
        return indices, distances[indices]
