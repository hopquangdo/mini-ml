import numpy as np
from .base import BaseLoss


class MSELoss(BaseLoss):
    def _compute(self):
        return np.mean((self.y_pred - self.y_true) ** 2)

    def _compute_grad(self):
        return 2 * (self.y_pred - self.y_true) / self.y_true.shape[0]


class MAELoss(BaseLoss):
    def _compute(self):
        return np.mean(np.abs(self.y_pred - self.y_true))

    def _compute_grad(self):
        return np.sign(self.y_pred - self.y_true) / self.y_true.shape[0]


class BCELoss(BaseLoss):
    def _compute(self):
        self.y_pred = np.clip(self.y_pred, 1e-8, 1 - 1e-8)
        return -np.mean(
            self.y_true * np.log(self.y_pred) +
            (1 - self.y_true) * np.log(1 - self.y_pred)
        )

    def _compute_grad(self):
        return (self.y_pred - self.y_true) / (
                self.y_pred * (1 - self.y_pred) * self.y_true.shape[0]
        )
