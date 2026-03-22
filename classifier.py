import numpy as np

def classify(perceptrons, x):
    # calculate the score for each perceptron using the score method
    scores = [p.score(x) for p in perceptrons]

    # return the index of the perceptron with the highest score, which corresponds to the predicted class
    return np.argmax(scores)

def classify_detailed(perceptrons, x):
    # returns all digits where predict() == 1, scores for all, and the best match
    predictions = []
    scores = []

    for i, p in enumerate(perceptrons):
        pred = p.predict(x)
        score = p.score(x)

        if pred == 1:
            predictions.append(i)

        scores.append(score)

    best = np.argmax(scores)
    
    return predictions, scores, best