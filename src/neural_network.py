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
    
    def run(self, array: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            array = layer.run(array)
        return array 

    def forward_propagation(self, array: np.ndarray) -> tuple[np.ndarray]:
        all_outputs = []
        for layer in self.layers:
            array = layer.run(array)
            all_outputs.append(array) 
        return all_outputs 

    def back_propagation_step(self, input: np.ndarray, correct_output: np.ndarray, learning_rate: float) -> None:
        data_size = input.shape[1]
        one_vector = np.ones(shape=(data_size, 1))

        # one batch revolution
        all_outputs = self.forward_propagation(input)
        delta = all_outputs[-1] - correct_output

        for i in range(len(self.dim)-2):
            j = -(i+1)
            dMatrix = (1 / data_size) * delta @ all_outputs[j].T
            dBias = (1 / data_size) * delta @ one_vector
            delta = self.layers[j].matrix.T @ delta * sigmoid_derivative_from_value(all_outputs[j-1])
            self.layers[j].set_matrix(self.layers[j].matrix - learning_rate * dMatrix)
            self.layers[j].set_bias( self.layers[j].bias - learning_rate * dBias)
        
    def train(self, input: np.ndarray, correct_output: np.ndarray, learning_rate: float, iterations: int) -> None:
        for i in range(iterations):
            self.back_propagation_step(input, correct_output, learning_rate)
        

    
    
    