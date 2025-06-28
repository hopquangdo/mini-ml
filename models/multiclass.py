import numpy as np
from models.core import BaseSupervisedModel


class OneVsRestClassifier(BaseSupervisedModel):
    def __init__(self, base_model_class, **model_kwargs):
        self.base_model_class = base_model_class
        self.model_kwargs = model_kwargs
        self.models = []
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = []

        for c in self.classes:
            binary_y = np.zeros_like(y)
            binary_y[y == c] = 1
            model = self.base_model_class(**self.model_kwargs)
            model.fit(X, binary_y)
            self.models.append(model)

    def predict(self, X):
        probs = np.array([model.predict_proba(X) for model in self.models])
        preds = np.argmax(probs, axis=0)
        return self.classes[preds]


class OneVsOneClassifier(BaseSupervisedModel):
    def __init__(self, base_model_class, **model_kwargs):
        self.base_model_class = base_model_class
        self.model_kwargs = model_kwargs
        self.models = []
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = []

        for i in range(len(self.classes)):
            for j in range(i + 1, len(self.classes)):
                class_1 = self.classes[i]
                class_2 = self.classes[j]

                idx = np.where((y == class_1) | (y == class_2))[0]
                X_pair = X[idx]
                y_pair = y[idx]

                binary_y = np.where(y_pair == class_1, 0, 1)

                model = self.base_model_class(**self.model_kwargs)
                model.fit(X_pair, binary_y)

                self.models.append((model, class_1, class_2))

    def predict(self, X):
        n_samples = X.shape[0]
        votes = np.zeros((n_samples, len(self.classes)))

        for model, class_1, class_2 in self.models:
            preds = model.predict(X)

            for i in range(n_samples):
                predicted_class = class_1 if preds[i] == 0 else class_2
                class_index = np.where(self.classes == predicted_class)[0][0]
                votes[i, class_index] += 1

        final_preds = np.argmax(votes, axis=1)
        return self.classes[final_preds]


class SoftmaxClassifier(BaseSupervisedModel):
    def __init__(self, lr=0.01, max_iter=1000):
        self.lr = lr
        self.max_iter = max_iter
        self.W = None
        self.b = None
        self.classes = []

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        y_one_hot = np.zeros((n_samples, n_classes))
        for idx, label in enumerate(self.classes):
            y_one_hot[y == label, idx] = 1

        self.W = np.zeros((n_features, n_classes))
        self.b = np.zeros((1, n_classes))

        for _ in range(self.max_iter):
            logits = X @ self.W + self.b
            probs = self._softmax(logits)

            grad_W = (1 / n_samples) * X.T @ (probs - y_one_hot)
            grad_b = (1 / n_samples) * np.sum(probs - y_one_hot, axis=0, keepdims=True)

            self.W -= self.lr * grad_W
            self.b -= self.lr * grad_b

    def predict_proba(self, X):
        logits = X @ self.W + self.b
        return self._softmax(logits)

    def predict(self, X):
        probs = self.predict_proba(X)
        class_indices = np.argmax(probs, axis=1)
        return self.classes[class_indices]
