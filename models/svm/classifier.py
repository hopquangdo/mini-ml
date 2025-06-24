import numpy as np
from models.svm.base import BaseSVM


class SVC(BaseSVM):
    def fit(self, X, y):
        y = np.where(y <= 0, -1, 1)
        n_samples, n_features = X.shape
        self._init_weights(n_features)

        for _ in range(self.max_iter):
            for i in range(n_samples):
                condition = y[i] * (X[i] @ self.w + self.b) < 1
                if condition:
                    self.w -= self.lr * (self.w - self.C * y[i] * X[i])
                    self.b += self.lr * self.C * y[i]
                else:
                    self.w -= self.lr * self.w

    def predict_proba(self, X):
        score = self.decision_function(X)
        return (score - score.min()) / (score.max() - score.min() + 1e-8)

    def predict(self, X):
        return np.sign(self.decision_function(X))
