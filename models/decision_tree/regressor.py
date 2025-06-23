import numpy as np
from models.decision_tree.base import BaseDecisionTree

class DecisionTreeRegressor(BaseDecisionTree):
    """
    Cây quyết định hồi quy.
    Dùng phương sai (variance) để đánh giá chất lượng chia.
    """

    def _impurity_score(self, y_left, y_right):
        """
        Tính tổng phương sai (variance) có trọng số sau khi chia nhánh.

        Parameters:
        - y_left: mảng nhãn bên trái
        - y_right: mảng nhãn bên phải

        Returns:
        - Giá trị tổng phương sai sau khi chia
        """
        n = len(y_left) + len(y_right)
        return (len(y_left) * np.var(y_left) + len(y_right) * np.var(y_right)) / n

    def _leaf_value(self, y):
        """
        Trả về trung bình của nhãn tại node lá (giá trị dự đoán).

        Parameters:
        - y: mảng nhãn tại node

        Returns:
        - Trung bình của y
        """
        return np.mean(y)
