import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from base import BaseSupervisedModel, BaseUnsupervisedModel


def evaluate_supervised(model: BaseSupervisedModel, X_train, y_train, X_test, y_test):
    """
    Đánh giá mô hình supervised (classification hoặc regression).
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    if is_classification(y_test):
        acc = accuracy_score(y_test, y_pred)
        print(f"✅ Accuracy: {acc:.4f}")
        return acc
    else:
        mse = mean_squared_error(y_test, y_pred)
        print(f"✅ MSE: {mse:.4f}")
        return mse


def evaluate_unsupervised(model: BaseUnsupervisedModel, X):
    """
    Đánh giá mô hình unsupervised (ví dụ: clustering).
    """
    model.fit(X)
    y_pred = model.predict(X)
    print(f"✅ Dự đoán (cluster labels): {np.unique(y_pred)}")
    return y_pred


def is_classification(y):
    """
    Heuristic: nếu nhãn là số nguyên rời rạc thì là classification.
    """
    return np.issubdtype(y.dtype, np.integer) and len(np.unique(y)) < 20
