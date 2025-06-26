import numpy as np
from models.cluster import KMeans
from models.base import BaseSupervisedModel


class KMeansRegressor(KMeans, BaseSupervisedModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.value_map = {}

    def fit(self, X, y=None):
        super().fit(X)
        labels = super().predict(X)

        for cluster_id in range(self.n_clusters):
            cluster_values = y[labels == cluster_id]
            print(cluster_values)
            if len(cluster_values) > 0:
                self.value_map[cluster_id] = np.mean(cluster_values)
            else:
                self.value_map[cluster_id] = 0.0

    def predict(self, X):
        cluster_ids = super().predict(X)
        return np.array([self.value_map[int(cid)] for cid in cluster_ids])

if __name__ == '__main__':
    X_train = np.array([
        [1, 2],
        [1, 3],
        [0, 2],
        [8, 9],
        [9, 8],
        [8, 8],
        [100, 100]
    ])

    y_train = np.array([10.0, 12.0, 11.0, 50.0, 52.0, 51.0, 1000.0])

    model = KMeansRegressor(n_clusters=3, max_iter=100)
    model.fit(X_train, y_train)
