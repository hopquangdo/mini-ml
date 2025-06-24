from models.svm.base import BaseSVM


class SVR(BaseSVM):
    def __init__(self, epsilon=0.1):
        super().__init__()
        self.epsilon = epsilon

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._init_weights(n_features)

        for _ in range(self.max_iter):
            for i in range(n_samples):
                y_pred = X[i] @ self.w + self.b
                error = y_pred - y[i]

                if error > self.epsilon:
                    self.w -= self.lr * (self.w + self.C * X[i])
                    self.b -= self.lr * self.C
                elif error < -self.epsilon:
                    self.w -= self.lr * (self.w + self.C * (-X[i]))
                    self.b += self.lr * self.C
                else:
                    self.w -= self.lr * self.w

    def predict(self, X):
        return self.decision_function(X)
