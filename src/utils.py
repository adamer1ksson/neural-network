import numpy as np
import csv

def sigmoid(vec: np.ndarray):
    log_vec = -np.logaddexp(0, -vec)
    return np.exp(log_vec)

def sigmoid_derivative_from_value(vec: np.ndarray):
    return vec * (1-vec)

def softmax(vec: np.ndarray):
    return np.exp(vec) / sum(np.exp(vec))

def value_to_vector(value: int) -> np.ndarray:
    vec = np.zeros(shape=(10,1))
    vec[value, 0] = 1
    return vec

def create_batches(data: csv.DictReader, max_data_points: int, batch_size: int) -> list[tuple[np.ndarray]]:
    """Specific to the MNIST dataset"""
    batches: list[tuple[np.ndarray]] = []

    correct_vectors: list[np.ndarray] = []
    data_vectors: list[np.ndarray] = []

    correct_matrix = 0
    data_matrix = 0

    for i, row in enumerate(data):
        if i+1 > max_data_points:
            break
    
        correct_vectors.append(value_to_vector(int(row[0])))
        data_vectors.append(np.array([[float(x)] for x in row[1:]]))

        if (i+1) % batch_size == 0:
            correct_matrix = np.column_stack(tuple(correct_vectors))
            data_matrix = np.column_stack(tuple(data_vectors))
            batches.append((data_matrix, correct_matrix))
            data_vectors = []
            correct_vectors = []
            correct_matrix = 0
            data_matrix = 0
    return batches