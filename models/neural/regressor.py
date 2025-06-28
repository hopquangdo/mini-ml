from models.neural.base import BaseNeuralNetwork


class RegressionNeuralNetwork(BaseNeuralNetwork):
    def predict(self, X):
        return self.forward(X)
