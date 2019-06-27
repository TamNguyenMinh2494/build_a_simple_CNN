from .layer import Layer
import numpy as np

class FCLayer(Layer):
    def __init__(self, input_shape, output_shape):
        """
        :param input_shape: (1,3)
        :param output_shape: (1,4)
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = np.random.rand(input_shape[1], output_shape[1]) - 0.5
        self.bias = np.random.rand(1, output_shape[1]) - 0.5
    def forward_propagation(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        current_layer_error = np.dot(output_error, self.weights.T) # self.weights : (3,4); output_error : (1,4) => chuyển vi5self.weights
        dweight = np.dot(self.input.T, output_error) # self.input: (1,3), output_error: (1,4) => chuyển vị self.input

        self.weights -= dweight*learning_rate #update weights
        self.bias -= output_error*learning_rate #update bias

        return current_layer_error