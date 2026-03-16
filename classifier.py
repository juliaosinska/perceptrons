import numpy as np

def classify(perceptrons, x):

    scores = []

    for p in perceptrons:

        s = np.dot(p.weights, x) + p.bias

        scores.append(s)

    return np.argmax(scores)