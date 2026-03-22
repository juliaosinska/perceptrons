import os
import numpy as np
import random
from PIL import Image
from perceptron import Perceptron
import pickle
from classifier import classify
from config import (RANDOM_SEED, WIDTH, HEIGHT, INPUT_SIZE, EPOCHS,
                   NOISE_PROBABILITY, TRAINING_DATA_FOLDER, MODEL_FILE)

# set random seeds for reproducible results
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

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

def add_noise(vec, prob=NOISE_PROBABILITY):
    noisy = vec.copy()
    for i in range(len(noisy)):
        if random.random() < prob:
            noisy[i] = 1 - noisy[i]

    return noisy


if __name__ == "__main__":
    # load training data
    X, y = load_dataset(TRAINING_DATA_FOLDER)
    print(f"Loaded {len(X)} training samples")
    
    # initialize one perceptron per digit
    perceptrons = [Perceptron(INPUT_SIZE) for _ in range(10)]
    
    print(f"Training for {EPOCHS} epochs with Pocket Learning Algorithm...")
    
    for epoch in range(EPOCHS):
        # train each perceptron on all samples
        for x, label in zip(X, y):
            # add noise to prevent overfitting
            x_noisy = add_noise(x)
            
            for digit in range(10):
                target = 1 if label == digit else 0
                perceptrons[digit].train(x_noisy, target)
        
        # Pocket Learning Algorithm: evaluate and store best weights
        for digit in range(10):
            y_binary = np.array([1 if label == digit else 0 for label in y])
            perceptrons[digit].update_pocket(X, y_binary)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} completed")
    
    for digit in range(10):
        perceptrons[digit].restore_best()
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(perceptrons, f)
    print(f"Model saved to {MODEL_FILE}")
    
    # evaluate training accuracy
    correct = 0
    
    for x, label in zip(X, y):
        pred = classify(perceptrons, x)
        if pred == label:
            correct += 1
    
    accuracy = correct / len(X)
    print(f"Training accuracy: {accuracy:.2%}")