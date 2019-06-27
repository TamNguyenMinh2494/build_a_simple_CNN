from network.network import Network
from layers.FCLayers import FCLayer
from layers.Activation_layer import activationLayer

import numpy as np
def reLu(z):
    """
    :param z: numpy array
    :return: return 0 if z <= 0
            return z if z >0
    [1, -3, 9, -7] --> reLu() --> [1, 0, 9, 0]
    """
    return np.maximum(0,z)
def reLu_prime(z):
    """
    :param z: numpy array
    :return: 1 if z > 0, 0 if z < 0

    """
    z[z < 0] = 0
    z[z > 0] = 1
    return z

def loss(y_true, y_pred):
    return 0.5*(y_pred - y_true)**2

def loss_prime(y_true, y_pred):
    return y_pred - y_true

x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

net = Network()
net.add(FCLayer((1,2), (1,3)))
net.add(activationLayer((1,3), (1,3), reLu, reLu_prime))
net.add(FCLayer((1,3), (1,1)))
net.add(activationLayer((1,1), (1,1), reLu, reLu_prime))

net.setup_lossFunction(loss, loss_prime)
net.fit(x_train, y_train, epochs=1000, learning_rate=0.01)

out = net.predict([[0,1]])

print(out)