import numpy as np
from collections import Counter
from models.tree.base import BaseDecisionTree


class DecisionTreeClassifier(BaseDecisionTree):
    def __init__(self, criterion="gini", **kwargs):
        super().__init__(**kwargs)
        self.criterion = criterion

    def _impurity_score(self, y_left, y_right, y_full=None):
        n = len(y_left) + len(y_right)
        if self.criterion == "gini":
            return (len(y_left) * self._gini(y_left) + len(y_right) * self._gini(y_right)) / n
        weighted_entropy = (len(y_left) * self._entropy(y_left) + len(y_right) * self._entropy(y_right)) / n
        return self._entropy(y_full) - weighted_entropy

    def _gini(self, y):
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        return 1.0 - np.sum(probs ** 2)

    def _entropy(self, y):
        y = np.asarray(y).ravel()
        if len(y) == 0:
            return 0.0
        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _leaf_value(self, y):
        return Counter(y).most_common(1)[0][0]
