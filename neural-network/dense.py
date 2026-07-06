import numpy as np
from layer import Layer

class Dense(Layer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.weight = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        # Z = WX + B
        return np.dot(self.weight, self.input) + self.bias

    def backward(self, output_gradient: np.ndarray, learn_rate: float) -> np.ndarray:
        weight_gradient = np.dot(output_gradient, self.input.T)
        bias_gradient = output_gradient

        self.weight -= learn_rate * weight_gradient
        self.bias -= learn_rate * bias_gradient

        return np.dot(self.weight.T, output_gradient)