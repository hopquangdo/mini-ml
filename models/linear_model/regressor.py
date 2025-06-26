import numpy as np
from models.linear_model.base import BaseLinearModel


class LinearRegression(BaseLinearModel):
    def fit(self, X, y):
        X_bias = self._add_bias(X)
        # Nghiệm tối ưu hóa MSE: w = (X^T X)^-1 X^T y
        self.weights = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y

    def predict(self, X):
        X_bias = self._add_bias(X)
        return X_bias @ self.weights
