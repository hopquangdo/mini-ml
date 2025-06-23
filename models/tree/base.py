"""/models/tree.py"""

import numpy as np
from models.base import BaseSupervisedModel

class BaseDecisionTree(BaseSupervisedModel):
    """
    Lớp cơ sở cho cây quyết định (dùng chung cho cả hồi quy và phân loại).
    Không dùng thư viện ngoài trừ numpy.
    """

    def __init__(self, max_depth=3, min_samples_split=2):
        """
        Khởi tạo cây với tham số giới hạn.

        Parameters:
        - max_depth: int, độ sâu tối đa của cây
        - min_samples_split: int, số lượng mẫu tối thiểu để tiếp tục chia
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    class Node:
        """
        Lớp con đại diện cho một nút trong cây (có thể là node trung gian hoặc node lá).
        """

        def __init__(self, depth, value=None, feature_index=None, threshold=None, left=None, right=None):
            """
            Parameters:
            - depth: int, độ sâu của node trong cây
            - value: giá trị dự đoán nếu là node lá
            - feature_index: chỉ số feature được chọn để chia
            - threshold: ngưỡng chia
            - left, right: cây con bên trái và phải
            """
            self.depth = depth
            self.value = value
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right

    def fit(self, X, y):
        """
        Huấn luyện cây quyết định trên dữ liệu huấn luyện.

        Parameters:
        - X: ndarray, dữ liệu đầu vào (n_samples x n_features)
        - y: ndarray, nhãn đầu ra
        """
        self.root = self._build_tree(X, y, depth=0)

    def predict(self, X):
        """
        Dự đoán đầu ra cho nhiều mẫu đầu vào.

        Parameters:
        - X: ndarray, tập dữ liệu đầu vào

        Returns:
        - y_pred: ndarray, dự đoán tương ứng
        """
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

    def _build_tree(self, X, y, depth):
        """
        Xây dựng cây quyết định một cách đệ quy.

        Parameters:
        - X: ndarray, tập dữ liệu tại node hiện tại
        - y: ndarray, nhãn tương ứng
        - depth: int, độ sâu hiện tại

        Returns:
        - Node đã được xây dựng
        """
        n_samples, n_features = X.shape

        # Điều kiện dừng: hết độ sâu hoặc dữ liệu quá ít
        if (depth >= self.max_depth) or (n_samples < self.min_samples_split):
            return self.Node(depth, value=self._leaf_value(y))

        best_feat, best_thresh, best_score = None, None, float('inf')

        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_mask = X[:, feature_index] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                score = self._impurity_score(y[left_mask], y[right_mask])
                if score < best_score:
                    best_score = score
                    best_feat = feature_index
                    best_thresh = threshold

        # Nếu không tìm được cách chia nào tốt → trả về node lá
        if best_feat is None:
            return self.Node(depth, value=self._leaf_value(y))

        # Chia dữ liệu theo best feature + threshold
        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        left_node = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_node = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return self.Node(
            depth=depth,
            value=None,
            feature_index=best_feat,
            threshold=best_thresh,
            left=left_node,
            right=right_node
        )

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
