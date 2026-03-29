# Configuration file for Perceptron Digit Classifier

# random seeds for reproducible results
RANDOM_SEED = 42

# grid dimensions
WIDTH = 5
HEIGHT = 7
INPUT_SIZE = WIDTH * HEIGHT  # 35

# training parameters
LEARNING_RATE = 0.1
EPOCHS = 100
NOISE_PROBABILITY = 0.05 

# file paths
TRAINING_DATA_FOLDER = "training_set"
MODEL_FILE = "perceptrons.pkl"

# GUI parameters
CELL_SIZE = 50
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 450