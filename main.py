from src.neural_network import NeuralNetwork
import numpy as np

def main():
    nn = NeuralNetwork([4, 3, 2])
    input = np.array([1, 2, 3, 4])
    res = nn.forward_propagation(input)
    print(res)



if __name__=="__main__":
    main()