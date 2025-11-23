import numpy as np
import random as random
import math as math

class Layer:
    def __init__(self, dimensions: tuple[int], func, total_dimension: tuple[int]):
        self.dim = (dimensions[1], dimensions[0]) # rows,cols
        self.func = func

        self.matrix = np.zeros(shape=self.dim)
        self.bias = np.zeros(self.dim[0])

        # Initialize matrix values
        for i in range(self.dim[0]):
            for j in range(self.dim[1]):
                value = random.uniform(-1, 1) * math.sqrt( 6 / sum(total_dimension) )
                self.matrix[i, j] = value

    def set_matrix(self, new_matrix: np.ndarray) -> None:
        self.matrix = new_matrix
    
    def set_bias(self, new_bias: np.ndarray) -> None:
        self.bias = new_bias

    def run(self, array: np.ndarray) -> np.ndarray:
        return self.func(self.matrix @ array + self.bias)