from collections import Counter
from models.neighbors.base import BaseKNN


class KNeighborsClassifier(BaseKNN):
    def _predict_one(self, x):
        nearest = self._get_k_nearest_indices(x)
        top_k_labels = self.y_train[nearest]
        return Counter(top_k_labels).most_common(1)[0][0]
