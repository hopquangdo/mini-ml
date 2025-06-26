import numpy as np
from collections import Counter
from models.neighbors.base import BaseKNN


class KNeighborsClassifier(BaseKNN):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _predict_one(self, x):
        indices, distances = self._get_k_nearest(x)
        labels = self.y_train[indices]

        if self.weights == 'uniform':
            return Counter(labels).most_common(1)[0][0]

        weights = self._compute_weights(distances, labels)

        unique_labels = np.unique(labels)
        label_weights = {label: np.sum(weights[labels == label]) for label in unique_labels}

        return max(label_weights.items(), key=lambda item: item[1])[0]


if __name__ == '__main__':
    model = KNeighborsClassifier()