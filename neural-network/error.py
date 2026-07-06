import numpy as np

# Calculate error
def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.power(y_true - y_pred, 2))

# Derivative of mean squared error function (dE/dY)
def mse_prime(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return 2 * (y_pred - y_true) / np.size(y_true)