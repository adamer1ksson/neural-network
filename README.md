# Learning the mathematics behind neural networks

## Goal with the project

My goal with this project was to understand **the math behind neural networks**. In order to acctually understand how neural networks function, *I have not* used any libraries, such as Tensorflow or Pytorch, but instead only Numpy. Down below follows a simple description of the classes in the directory, and the results of the model. 

## Results
On the MNIST dataset ([Link](https://github.com/phoebetronic/mnist)), the model achieves an accuracy of approximately 92%.
This is not a very impressive accuracy, however it is to be expected since the neural network **only uses dense layers** at the moment. An improvement, that will be implementet in the future, would be to add convolution layers. This would probably greatly increse the accuracy as well as lower the training time of the model. 

## Classes
The neural network is built out of two main classes, ```NeuralNetwork``` and ```Layer```. Layer is a hidden class that is not used when utalizing the library for training a model. Below follows an example how to use the class ```NeuralNetwork``` and train it on data.

```python
import numpy as np
from src.neural_network import NeuralNetwork

# Say you have vectors of length 100 as input data, that you want to
# classify into 10 categories with 1 hidden layer of size 20, the code is:
NN = NeuralNetwork([100, 20, 10]) 

# Assume that you have created batches of following type:
batches: list[tuple[np.ndarray]]

# Train model with data:
learning_rate = 0.01
epochs = 100
NN.train(batches, learning_rate, epochs)
```
After you have trained the network, you can simply test on some test batches by running the following code:
```python
batch_size = 100 # this is passed to the test function for simplicity
procentage_correct = NN.test(test_batches, batch_size)
print(procentage_correct)
```


