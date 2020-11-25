import numpy as np

# Constants
DATA_SIZE = 100
NUM_FEATURES = 2
LEARNING_RATE = 0.8

# Generate dataset
X = (np.random.rand(DATA_SIZE, 2) - 0.5) * 500
if NUM_FEATURES != 2:
    error ("Wrong number of features")
y = 1* ((X.T[0] * (2*X.T[1]) - 2) > 0)
print (X)
print (y)

# Weights
weights = np.random.rand(2, 1)
print (weights)

# Learn
o = np.multiply(weights.T, X)
# FIXME how to calculate E when there is a square
