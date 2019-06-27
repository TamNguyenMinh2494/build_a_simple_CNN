class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None
    def add(self, layer):
        self.layers.append(layer)
    def setup_lossFunction(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime
    def predict(self, input):
        """
        :param input:
        :return:
        """

        result = []
        n = len(input)
        for i in range(n):
            output = input[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)
        return result
    def fit(self, x_train, y_train, learning_rate, epochs):
        # training function
        n = len(x_train)
        for i in range(epochs):
            err = 0 # tính tổng lỗi
            for j in range(n):
                #lan truyền tiến
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                #tính lỗi từng sample
                err += self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output) #lỗi lớp cuối cùng khi chuẩn bị lan truyền ngược
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            err /= n

            print('epoch: %d/%d \t err = %f' % (i, epochs, err))
