# models/linear/classifier.py

import numpy as np
from models.linear_model.base import BaseLinearModel
from models.utils import ensure_numpy

class LinearClassification(BaseLinearModel):
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def predict(self, X):
        X_bias = self._add_bias(ensure_numpy(X))
        logits = X_bias @ self.weights
        probs = 1 / (1 + np.exp(-logits))
        return (probs >= self.threshold).astype(int)
