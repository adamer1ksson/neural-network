import numpy as np
import math as math

def sigmoid(vec: np.ndarray):
    return 1 / (1+ np.exp(-vec))

def sigmoid_derivative_from_value(vec: np.ndarray):
    return vec * (1-vec)

def softmax(vec: np.ndarray):
    return np.exp(vec) / sum(np.exp(vec))