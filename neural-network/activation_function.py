import numpy as np
from activation import Activation

class Sigmoid(Activation):
    def __init__(self):
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        sigmoid_prime = lambda y: sigmoid(y) * (1 - sigmoid(y))
        super().__init__(sigmoid, sigmoid_prime)