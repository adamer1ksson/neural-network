from src.neural_network import NeuralNetwork
import numpy as np


# mnist dataset link github https://github.com/phoebetronic/mnist 

def main():
    nn = NeuralNetwork([4, 2, 2])
    input = np.array([[1], [2], [3], [4]])
    output = np.array([[1], [0]])
    nn.train(input, output, 0.1, 100)
    res = nn.run(input)
    print(res)
    



if __name__=="__main__":
    main()