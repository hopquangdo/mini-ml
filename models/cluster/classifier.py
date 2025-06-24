from scipy.stats import mode
import numpy as np
from .base import KMeans
from models.base import BaseSupervisedModel


class KMeansClassifier(KMeans, BaseSupervisedModel):
    def __init__(self, n_clusters=3, max_iter=100, tol=1e-4):
        super().__init__(n_clusters, max_iter, tol)
        self.label_map = {}

    def fit(self, X, y):
        super().fit(X)
        cluster_ids = super().predict(X)

        for cluster_id in range(self.n_clusters):
            cluster_points = y[cluster_ids == cluster_id]
            if len(cluster_points) > 0:
                self.label_map[cluster_id] = mode(cluster_points, keepdims=True).mode.item()
            else:
                self.label_map[cluster_id] = -1

    def predict(self, X):
        cluster_ids = super().predict(X)
        return np.array([self.label_map[cid] for cid in cluster_ids])
