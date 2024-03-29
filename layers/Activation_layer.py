from .layer import Layer
class activationLayer(Layer):
    def __init__(self, input_shape, output_shape, activation, activation_prime):
        """
        :param input_shape: là mảng đầu vào vd (1,4)
        :param output_shape: mảng
        :param activation: hàm active
        :param activation_prime: đạo hàm hàm active
        """
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input):

        self.input = input
        self.output = self.activation(input)
        return self.output

    def backward_propagation(self, output_error, learning_rate):
        return self.activation_prime(self.input)*output_error