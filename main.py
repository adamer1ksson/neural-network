import numpy as np
import csv

from src.neural_network import NeuralNetwork
from src.utils import create_batches

# MNIST dataset link github https://github.com/phoebetronic/mnist 

def main():
    nn = NeuralNetwork([784, 10, 10, 10])    

    print("Creating batches")
    with open("mnist_train.csv", encoding="utf8") as file:
        data = csv.reader(file, delimiter=",")
        batches: list[tuple[np.ndarray]] = create_batches(data, 60000, 100)        
    
    training_batches = batches[:-50]
    test_batches = batches[-50:]
    print(len(training_batches), len(test_batches))
    
    epochs = 250
    print("Begin training")
    nn.train(batches, 0.02, epochs)

    print("Begin testing")
    procentage_correct = nn.test(test_batches, 100)
    print(procentage_correct)

if __name__=="__main__":
    main()