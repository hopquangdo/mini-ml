import numpy as np

class DenseLayer:
    def __init__(self, input_size, output_size, activation):
        self.W = np.random.randn(input_size, output_size)
        self.b = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, X):
        self.input = X
        self.z = X @ self.W + self.b
        self.a = self.activation.forward(self.z)
        return self.a

    def backward(self, grad_output, learning_rate):
        grad_z = grad_output * self.activation.backward(self.a)
        grad_W = self.input.T @ grad_z
        grad_b = grad_z.sum(axis=0, keepdims=True)
        grad_input = grad_z @ self.W.T

        self.W -= learning_rate * grad_W
        self.b -= learning_rate * grad_b
        return grad_input