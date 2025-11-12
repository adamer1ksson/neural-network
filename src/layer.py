import numpy as np

class Layer:
    def __init__(self, dimensions: tuple[int], func):
        self.dim = dimensions
        self.func = func

        self.matrix = np.zeros(shape=(self.dim[1], self.dim[0]))
        self.bais = np.zeros(shape=(self.dim[0], 1))

    def run(self, array: np.ndarray):
        pass