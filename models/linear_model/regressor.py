import numpy as np

class LinearRegression:
    """
    Mô hình Linear Regression sử dụng công thức nghiệm đóng (closed-form).
    """

    def __init__(self):
        self.weights = None

    def fit(self, X, y):
        """
        Huấn luyện mô hình bằng công thức nghiệm đóng: w = (X^T X)^-1 X^T y
        """
        X_bias = np.c_[np.ones((X.shape[0],)), X]  # Thêm bias (1) vào đầu
        self.weights = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y

    def predict(self, X):
        """
        Dự đoán đầu ra với mô hình đã huấn luyện.
        """
        X_bias = np.c_[np.ones((X.shape[0],)), X]
        return X_bias @ self.weights
