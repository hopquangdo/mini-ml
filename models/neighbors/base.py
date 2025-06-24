import numpy as np
from models.base import BaseSupervisedModel
from models.utils import ensure_numpy


class BaseKNN(BaseSupervisedModel):
    """
    Lớp cơ sở dùng chung cho KNN Classification và Regression.
    """

    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """
        Lưu dữ liệu huấn luyện.
        """
        self.X_train = ensure_numpy(X)
        self.y_train = ensure_numpy(y)

    def predict(self, X):
        """
        Dự đoán đầu ra cho tập dữ liệu X.
        """
        X = ensure_numpy(X)
        return np.array([self._predict_one(x) for x in X])

    def _predict_one(self, x):
        """
        Dự đoán cho một mẫu đơn. Hàm này sẽ được override.
        """
        raise NotImplementedError("Subclass cần cài đặt _predict_one()")
