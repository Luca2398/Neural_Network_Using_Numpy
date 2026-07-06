import numpy as np
from sklearn.datasets import fetch_openml


mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')

X = mnist.data / 255.0
Y = mnist.target.astype(int)

X_formatted = [x.reshape(784, 1) for x in X]

def one_hot(y):
    encoded = np.zeros((10, 1))
    encoded[y] = 1.0
    return encoded

Y_formatted = [one_hot(y) for y in Y]
X_train, X_test = X_formatted[:60000], X_formatted[60000:]
Y_train, Y_test = Y_formatted[:60000], Y_formatted[60000:]


from layer import Layer
from dense import Dense
from activation import Activation
from activation_function import Sigmoid
from error import mse, mse_prime

network = [
    Dense(784, 128),
    Sigmoid(),
    Dense(256, 64),
    Sigmoid(),
    Dense(64, 10),
    Sigmoid()
]

epochs = 10000
learn_rate = 0.001

for e in range(epochs):
    error = 0
    for x, y in zip(X_train, Y_train):
        # FORWARD PROPAGATION
        output = x
        for layer in network:
            output = layer.forward(output)

        error += mse(y, output)

        # BACKWARD PROPAGATION
        gradient = mse_prime(y, output)

        for layer in reversed(network):
            gradient = layer.backward(gradient, learn_rate)

    error /= len(x)
    if e % 1000 == 0:
        print(f"Epoch {e}/{epochs} - Error: {error:.6f}")