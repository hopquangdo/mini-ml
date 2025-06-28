import numpy as np
from models.neural.core import Layer


class ReLU(Layer):
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, grad_output, learning_rate):
        return grad_output * (self.input > 0)


class Sigmoid(Layer):
    def forward(self, input):
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, grad_output, learning_rate):
        return grad_output * self.output * (1 - self.output)


class Tanh(Layer):
    def forward(self, input):
        self.output = np.tanh(input)
        return self.output

    def backward(self, grad_output, learning_rate):
        return grad_output * (1 - self.output ** 2)
