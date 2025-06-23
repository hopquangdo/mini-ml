import numpy as np
from collections import Counter
from models.decision_tree.base import BaseDecisionTree

class DecisionTreeClassifier(BaseDecisionTree):
    """
    Cây quyết định phân loại.
    Dùng chỉ số Gini để đánh giá chất lượng chia.
    """

    def _impurity_score(self, y_left, y_right):
        """
        Tính chỉ số Gini có trọng số sau khi chia nhánh.

        Parameters:
        - y_left: mảng nhãn bên trái
        - y_right: mảng nhãn bên phải

        Returns:
        - Giá trị Gini sau khi chia
        """
        n = len(y_left) + len(y_right)
        return (len(y_left) * self._gini(y_left) + len(y_right) * self._gini(y_right)) / n

    def _gini(self, y):
        """
        Tính chỉ số Gini impurity của một node.

        Parameters:
        - y: mảng nhãn

        Returns:
        - Gini impurity
        """
        counts = np.bincount(y)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _leaf_value(self, y):
        """
        Trả về nhãn xuất hiện nhiều nhất tại node lá.

        Parameters:
        - y: mảng nhãn tại node

        Returns:
        - Nhãn phổ biến nhất
        """
        return Counter(y).most_common(1)[0][0]
