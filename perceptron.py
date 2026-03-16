import numpy as np

class Perceptron:

    def __init__(self, input_size, learning_rate=0.1):

        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.lr = learning_rate

    def predict(self, x):

        s = np.dot(self.weights, x) + self.bias

        if s > 0:
            return 1
        else:
            return 0

    def train(self, x, target):

        prediction = self.predict(x)

        error = target - prediction

        self.weights += self.lr * error * x
        self.bias += self.lr * error