from typing import List
from models.core import BaseSupervisedModel
from models.neural.core.base import Layer


class NeuralNetwork(BaseSupervisedModel):
    def __init__(self, layers: List[Layer], loss):
        self.layers = layers
        self.loss = loss

    def forward(self, X):
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad, learning_rate):
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def fit(self, X, y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss_val = self.loss.forward(y_pred, y)
            grad = self.loss.backward()
            self.backward(grad, learning_rate)

        print(f"Final Loss after {epochs} epochs: {loss_val:.4f}")

    def predict(self, X):
        return self.forward(X)
