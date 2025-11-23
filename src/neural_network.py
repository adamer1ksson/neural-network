import numpy as np

from src.layer import Layer
from src.utils import sigmoid, sigmoid_derivative_from_value, softmax

class NeuralNetwork:
    def __init__(self, dimensions: list[int]):
        self.dim = dimensions
        self.total_dim = (dimensions[0], dimensions[-1])
        self.layers = []

        # initializing layers
        for i, el in enumerate(self.dim):
            if i == 0 or i == (len(self.dim) - 1):
                continue
            layer = Layer((self.dim[i-1], self.dim[i]), sigmoid, self.total_dim)
            self.layers.append(layer)
        
        # last layer with softmax
        last_layer = Layer((self.dim[-2],self.dim[-1]), softmax, self.total_dim)
        self.layers.append(last_layer)
    
    def forward_propagation(self, array: np.ndarray) -> tuple[np.ndarray]:
        all_outputs = []
        for layer in self.layers:
            array = layer.run(array)
            all_outputs.append(array) 
        return all_outputs 

    def train(self, input: np.ndarray, correct_output: np.ndarray, n: int, learning_rate: float):
        data_points = input.shape[1]

        # one revolution
        all_outputs = self.forward_propagation(input)
        delta = all_outputs[-1] - correct_output
        for i in range(len(self.dim)-1):
            j = -(i+1)
            dMatrix = delta @ all_outputs[j]
            dBias = delta
            delta = self.layers[j].matrix.T @ delta * sigmoid_derivative_from_value(all_outputs[j])
            self.layers[j].matrix -= learning_rate * dMatrix
            self.layers[j].bias -= learning_rate * dBias
            


        for i in range(n):
            pass


    
    
    