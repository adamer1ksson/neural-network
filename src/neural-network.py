import numpy as np

from src.layer import Layer
from src.utils import sigmoid, softmax

class NeuralNetwork:
    def __init__(self, dimensions: list[int]):
        self.dim = dimensions
        self.layers = []

        # initializing layers
        for i, el in enumerate(self.dim):
            if i == 0 or i == (len(self.dim) - 1):
                continue
            layer = Layer(el, self.dim[0], sigmoid)
            self.layers.append(layer)
        last_layer = Layer((self.dim[-1],self.dim[-2]), softmax)
        self.layers.append(last_layer)
    
    def forward_propagation(self, array: np.ndarray) -> np.ndarray:
        pass