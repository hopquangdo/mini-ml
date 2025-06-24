from models.base import BaseSupervisedModel
import numpy as np
from models.tree._tree import TreeBuilder


class BaseDecisionTree(BaseSupervisedModel):
    def __init__(self, max_depth=3, min_samples_split=2, use_all_thresholds=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.use_all_thresholds = use_all_thresholds

    def fit(self, X, y):
        builder = TreeBuilder(
            impurity_func=self._impurity_score,
            leaf_func=self._leaf_value,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            use_all_thresholds=self.use_all_thresholds
        )
        self.root = builder.build(X, y)

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        """
        Dự đoán cho một điểm dữ liệu duy nhất bằng cách đi qua cây.

        Parameters:
        - x: ndarray, một vector đặc trưng
        - node: Node, nút hiện tại đang xét

        Returns:
        - giá trị dự đoán tại node lá
        """
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)

    def _impurity_score(self, y_left, y_right):
        """
        Hàm trừu tượng tính độ nhiễu (impurity) của cách chia hiện tại.
        Override trong lớp con (hồi quy → variance, phân loại → Gini/entropy).

        Parameters:
        - y_left: các nhãn bên trái
        - y_right: các nhãn bên phải

        Returns:
        - giá trị impurity (càng thấp càng tốt)
        """
        raise NotImplementedError

    def _leaf_value(self, y):
        """
        Hàm trừu tượng tính giá trị dự đoán tại node lá.
        Override trong lớp con (mean với hồi quy, mode với phân loại).

        Parameters:
        - y: mảng nhãn tại node lá

        Returns:
        - giá trị dự đoán
        """
        raise NotImplementedError
