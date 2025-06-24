import numpy as np
from .base import KMeans
from models.base import BaseSupervisedModel


class KMeansRegressor(KMeans, BaseSupervisedModel):
    def fit(self, X, y):
        super().fit(X)
        labels = self.predict(X)

        self.value_map = {}
        for cluster_id in range(self.n_clusters):
            cluster_values = y[labels == cluster_id]
            if len(cluster_values) > 0:
                self.value_map[cluster_id] = np.mean(cluster_values)
            else:
                self.value_map[cluster_id] = 0.0  # fallback

    def predict(self, X):
        cluster_ids = super().predict(X)
        return np.array([self.value_map[cid] for cid in cluster_ids])
