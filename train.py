import os
import numpy as np
import random
from PIL import Image
from perceptron import Perceptron
import pickle
from classifier import classify

WIDTH = 5
HEIGHT = 7

def load_image(path):
    img = Image.open(path).convert("RGBA")
    data = np.array(img)
    alpha = data[:, :, 3]
    binary = (alpha > 0).astype(int)
    vector = binary.flatten()

    return vector

def load_dataset(folder):
    X = []
    y = []

    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            path = os.path.join(folder, filename)
            digit = int(filename.split("_")[0])
            vector = load_image(path)

            X.append(vector)
            y.append(digit)

    return np.array(X), np.array(y)

def add_noise(vec, prob=0.05):
    noisy = vec.copy()
    for i in range(len(noisy)):
        if random.random() < prob:
            noisy[i] = 1 - noisy[i]

    return noisy


X, y = load_dataset("training_set")

perceptrons = []

for i in range(10):
    perceptrons.append(Perceptron(35))

epochs = 50

for epoch in range(epochs):
    for x, label in zip(X, y):
        x_noisy = add_noise(x)
        for digit in range(10):
            target = 1 if label == digit else 0
            perceptrons[digit].train(x_noisy, target)
    
    # Pocket Learning Algorithm: after each epoch, check if current weights are better
    for digit in range(10):
        # create binary targets for this digit: 1 if label == digit, else 0
        y_binary = np.array([1 if label == digit else 0 for label in y])
        # update pocket with best weights found so far
        perceptrons[digit].update_pocket(X, y_binary)

# restore best weights found during training (Pocket Algorithm)
for digit in range(10):
    perceptrons[digit].restore_best()

# save the trained perceptrons to a file for later use in the GUI
with open('perceptrons.pkl', 'wb') as f:
    pickle.dump(perceptrons, f)

# add a simple evaluation of the training accuracy by classifying the training samples 
# and comparing to the true labels
correct = 0
total = len(X)
for x, label in zip(X, y):
    pred = classify(perceptrons, x)
    if pred == label:
        correct += 1

accuracy = correct / total
print(f"Training accuracy: {accuracy:.2f}")