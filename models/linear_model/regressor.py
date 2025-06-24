# models/linear/regression.py

import numpy as np
from models.linear_model.base import BaseLinearModel
from models.utils import ensure_numpy


class LinearRegression(BaseLinearModel):
    def predict(self, X):
        X_bias = self._add_bias(ensure_numpy(X))
        return X_bias @ self.weights
