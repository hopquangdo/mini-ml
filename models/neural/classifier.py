from models.neural.core import BaseNeuralNet


class ClassificationNeuralNet(BaseNeuralNet):
    def __init__(self, **kwargs):
        super(ClassificationNeuralNet, self).__init__(**kwargs)
