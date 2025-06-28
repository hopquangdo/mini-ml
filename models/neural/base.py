from models.core import BaseSupervisedModel


class BaseNeuralNetwork(BaseSupervisedModel):
    def __init__(self, layers, loss):
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
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss_val:.4f}")
