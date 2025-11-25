import numpy as np
import csv

from src.neural_network import NeuralNetwork


# mnist dataset link github https://github.com/phoebetronic/mnist 

def value_to_vector(value: int) -> np.ndarray:
    vec = np.zeros(shape=(10,1))
    vec[value, 0] = 1
    return vec

def main():
    nn = NeuralNetwork([784, 10, 10, 10])    

    with open("mnist_train.csv", encoding="utf8") as file:
        data = csv.reader(file, delimiter=",")

        batches: list[tuple[np.ndarray]] = []
        correct_vectors = []
        data_vectors = []
        correct_matrix = 0
        data_matrix = 0

        i = 1
        for row in data:
            if i % 100 == 0:
                correct_matrix = np.column_stack(tuple(correct_vectors))
                data_matrix = np.column_stack(tuple(data_vectors))
                batches.append((data_matrix, correct_matrix))
                correct_matrix = 0
                data_matrix = 0
                data_vectors = []
                correct_vectors = []
            if i == 10000:
                break
            correct_vectors.append(value_to_vector(int(row[0])))
            data_vectors.append(np.array([[float(x)] for x in row[1:]]))
            i += 1
        test_batch = batches.pop()
    epochs = 200
    for i in range(epochs):
        for batch in batches:
            # print(type(batch), type(batch[0]), type(batch[1]), batch[0].shape, batch[1].shape)
            nn.train(batch[0], batch[1], 0.01, 1)
    input_vector = np.array([[x] for x in batches[1][0][:,0]])
    print(nn.run(input_vector))
    print(value_to_vector(2).shape)
    print(batches[1][1][:,0])
    
    


        
    



    

    
    



if __name__=="__main__":
    main()