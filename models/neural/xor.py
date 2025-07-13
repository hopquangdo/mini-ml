import numpy as np
from scipy.special import expit as sigmoid

# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([[0], [1], [1], [0]])

# Hyperparameters
np.random.seed(42)
lr = 0.1
epochs = 1000

input_dim = 2  # số chiều input (XOR = 2)
hidden_dim = 8  # Số neuron ở hidden layer
output_dim = 1  # XOR chỉ có 1 output

W1 = np.random.randn(input_dim, hidden_dim)
print(f"W1: {W1}")
b1 = np.zeros((1, hidden_dim))
print(f"b1: {b1}")
W2 = np.random.randn(hidden_dim, output_dim)
print(f"W2: {W2}")
b2 = np.zeros((1, output_dim))
print(f"b2: {b2}")


def tanh(x):
    return np.tanh(x)


# Training loop
for epoch in range(epochs):
    # ----- Forward Pass -----
    z1 = X @ W1 + b1
    a1 = tanh(z1)
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)

    # ----- Compute Binary Cross-Entropy Loss -----
    loss = -np.mean(y * np.log(a2 + 1e-8) + (1 - y) * np.log(1 - a2 + 1e-8))

    # ----- Backward Pass -----
    dL_da2 = -(y / (a2 + 1e-8)) + (1 - y) / (1 - a2 + 1e-8)
    da2_dz2 = a2 * (1 - a2)
    dz2 = dL_da2 * da2_dz2

    dW2 = a1.T @ dz2
    db2 = np.sum(dz2, axis=0, keepdims=True)

    da1_dz1 = 1 - a1 ** 2  # derivative of tanh
    dz1 = (dz2 @ W2.T) * da1_dz1

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # ----- Gradient Descent Update -----
    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1

    # ----- Visualization on the last epoch -----
    if epoch == epochs - 1:
        print(f"dW1: {dW1}, db1: {db1}, dW2: {dW2}, db2: {db2}")
        print(f"Loss: {loss:.4f}")
        print("\nFinal output after training:")
        for i in range(len(X)):
            x1, x2 = X[i]
            raw = z2[i][0]
            prob = a2[i][0]
            print(f"Input: [{int(x1)}, {int(x2)}] → z2: {raw:.6f} → sigmoid(z2): {prob:.6f}")
