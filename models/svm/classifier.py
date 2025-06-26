import numpy as np
from models.svm.base import BaseSVM


class SVC(BaseSVM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        y = np.where(y == self.classes[0], -1, 1)
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
        return 1 / (1 + np.exp(-score))

    def predict(self, X):
        scores = self.decision_function(X)
        preds = []

        for score in scores:
            if score >= 0:
                preds.append(self.classes[1])
            else:
                preds.append(self.classes[0])

        return np.array(preds)


if __name__ == '__main__':
    X = np.array([
        [2, 3],
        [1, 5],
        [2, 8],
        [8, 8],
        [9, 5],
        [10, 2]
    ])

    y = np.array([3, 3, 3, 5, 5, 5])

    model = SVC(C=1.0, lr=0.01, max_iter=1000)

    model.fit(X, y)
    print(model.predict_proba(X))

    preds = model.predict(X)
    print("Predictions:", preds)
    print("True Labels:", y)

    acc = np.mean(preds == y)
    print(f"Accuracy: {acc:.2f}")
