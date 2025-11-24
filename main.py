import numpy as np
import csv

from src.neural_network import NeuralNetwork


# mnist dataset link github https://github.com/phoebetronic/mnist 

def value_to_vector(value: int) -> np.ndarray:
    vec = np.zeros(shape=(10,1))
    vec[value, 0] = 1

def main():
    nn = NeuralNetwork([784, 10, 10, 10])

    with open("mnist_train.csv", encoding="utf8") as file:
        data = csv.reader(file, delimiter=",")

        correct_vectors = []
        data_vectors = []

        i = 1
        for row in data:
            if i == 10000:
                break
            correct_vectors.append(value_to_vector(int(row[0])))
            data_vectors.append(np.array([[float(x)] for x in row[1:]]))
            i += 1
        
        correct_matrix = np.column_stack(tuple(correct_vectors))
        data_matrix = np.column_stack(tuple(data_vectors))
        
    



    

    
    



if __name__=="__main__":
    main()