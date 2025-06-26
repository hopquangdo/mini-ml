import numpy as np
from models.tree.base import BaseDecisionTree


class DecisionTreeRegressor(BaseDecisionTree):
    def __init__(self, criterion="mse", **kwargs):
        super().__init__(**kwargs)
        self.criterion = criterion

    def _impurity_score(self, y_left, y_right, y_full=None):
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right

        if self.criterion == "mse":
            mse_left = np.var(y_left) if n_left > 0 else 0
            mse_right = np.var(y_right) if n_right > 0 else 0
            return (n_left * mse_left + n_right * mse_right) / n_total

        mae_left = np.mean(np.abs(y_left - np.mean(y_left))) if n_left > 0 else 0
        mae_right = np.mean(np.abs(y_right - np.mean(y_right))) if n_right > 0 else 0
        return (n_left * mae_left + n_right * mae_right) / n_total

    def _leaf_value(self, y):
        return np.mean(y)
