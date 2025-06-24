import numpy as np
from models.base import BaseSupervisedModel


class BaseKNN(BaseSupervisedModel):
    """
    Lớp cơ sở dùng chung cho KNN Classification và Regression.
    """

    def __init__(self, k=3, distance_metric='euclidean'):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Lưu dữ liệu huấn luyện.
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """
        Dự đoán đầu ra cho tập dữ liệu X.
        """
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        """
        Dự đoán cho một mẫu đơn. Hàm này sẽ được override.
        """
        raise NotImplementedError("Subclass cần cài đặt _predict_one()")

    def _get_k_nearest_indices(self, x):
        if self.distance_metric == 'euclidean':
            distances = np.linalg.norm(self.X_train - x, axis=1)
        elif self.distance_metric == 'manhattan':
            distances = np.sum(np.abs(self.X_train - x), axis=1)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
        return np.argsort(distances)[:self.k]
