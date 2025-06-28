from models.base import BaseSupervisedModel


class BaseNeuralNet(BaseSupervisedModel):
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim=1,
                 learning_rate=0.1):
        self.lr = learning_rate
        