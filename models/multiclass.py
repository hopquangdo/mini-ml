import numpy as np
from models.base import BaseSupervisedModel


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
