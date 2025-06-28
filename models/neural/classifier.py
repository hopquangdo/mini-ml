from models.neural.base import BaseNeuralNetwork


class ClassificationNeuralNetwork(BaseNeuralNetwork):
    def __init__(self, threshold=0.5, **kwargs):
        super(ClassificationNeuralNetwork, self).__init__(**kwargs)
        self.threshold = threshold

    def predict(self, X):
        probs = self.forward(X)
        return (probs > self.threshold).astype(int)
