from collections import Counter
import numpy as np
from models.cluster import KMeans
from models.core.losses.base import BaseSupervisedModel


class KMeansClassifier(KMeans, BaseSupervisedModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.label_map = {}

    def fit(self, X, y=None):
        super().fit(X)
        cluster_ids = super().predict(X)

        for cluster_id in range(self.n_clusters):
            mask = (cluster_ids == cluster_id)
            cluster_points = y[mask]

            if len(cluster_points) > 0:
                self.label_map[cluster_id] = Counter(cluster_points).most_common(1)[0][0]
            else:
                self.label_map[cluster_id] = -1

    def predict(self, X):
        cluster_ids = super().predict(X)
        predictions = [self.label_map.get(cid, -1) for cid in cluster_ids]
        return np.array(predictions)


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
    y_train = np.array([0, 0, 0, 1, 1, 1, 3])

    model = KMeansClassifier(n_clusters=3, max_iter=100)
    model.fit(X_train, y_train)
    print(model.predict(X_train))
