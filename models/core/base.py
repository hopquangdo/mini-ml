class BaseSupervisedModel:
    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError


class BaseUnsupervisedModel:
    def fit(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
