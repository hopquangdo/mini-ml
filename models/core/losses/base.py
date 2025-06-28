from abc import ABC, abstractmethod


class BaseLoss(ABC):
    def forward(self, y_pred, y_true):
        """Tính giá trị loss"""
        self.y_pred = y_pred
        self.y_true = y_true
        return self._compute()

    def backward(self):
        """Tính gradient của loss theo y_pred"""
        return self._compute_grad()

    @abstractmethod
    def _compute(self):
        pass

    @abstractmethod
    def _compute_grad(self):
        pass
