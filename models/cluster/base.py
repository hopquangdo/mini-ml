import numpy as np
from models.core import BaseUnsupervisedModel


class KMeans(BaseUnsupervisedModel):
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, X):
        np.random.seed(42)
        indices = np.random.permutation(len(X))[:self.n_clusters]
        self.centroids = X[indices]

        for _ in range(self.max_iter):
            labels = self._assign_labels(X)

            new_centroids = np.array([
                X[labels == i].mean(axis=0) if len(X[labels == i]) > 0 else self.centroids[i]
                for i in range(self.n_clusters)
            ])

            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break

            self.centroids = new_centroids

    def predict(self, X):
        return self._assign_labels(X)

    def _assign_labels(self, X):
        distances = []
        for point in X:
            point_distances = np.linalg.norm(point - self.centroids, axis=1)
            distances.append(point_distances)
        return np.argmin(distances, axis=1)
