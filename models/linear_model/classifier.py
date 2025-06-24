import numpy as np
from models.linear_model.base import BaseLinearModel


class LogisticRegression(BaseLinearModel):
    def __init__(self, lr=0.01, max_iter=1000, threshold=0.5):
        super().__init__()
        self.lr = lr
        self.max_iter = max_iter
        self.threshold = threshold

    def fit(self, X, y):
        X_bias = self._add_bias(X)
        n_samples, n_features = X_bias.shape
        self.weights = np.zeros(n_features)

        for _ in range(self.max_iter):
            logits = X_bias @ self.weights
            probs = 1 / (1 + np.exp(-logits))
            gradient = X_bias.T @ (probs - y) / n_samples
            self.weights -= self.lr * gradient

    def predict_proba(self, X):
        """Predict Probability of X"""
        X_bias = self._add_bias(X)
        logits = X_bias @ self.weights
        return 1 / (1 + np.exp(-logits))

    def predict(self, X):
        probs = self.predict_proba(X)
        return (probs >= self.threshold).astype(int)
