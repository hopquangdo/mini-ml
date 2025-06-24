from collections import Counter
import numpy as np
from models.neighbors.base import BaseKNN

class KNeighborsClassifier(BaseKNN):
    """
    KNN cho phân loại.
    """

    def _predict_one(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        nearest = np.argsort(distances)[:self.k]
        top_k_labels = self.y_train[nearest]
        return Counter(top_k_labels).most_common(1)[0][0]
