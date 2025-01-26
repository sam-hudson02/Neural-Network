from utils.data import load_mnist_data, load_cifar_data
import numpy as np
from utils.utils import Activation, mse, mse_prime
from models.classify import Classifier
from layers.dense import Dense
from layers.activation import Activation as ActivationLayer
from layers.convolution import Convolution
from layers.reshape import Reshape
from models.nn import Network
from utils.activation import ReLU, Sigmoid


def network():
    x_test, y_test, x_train, y_train = load_mnist_data()
    layers = [
        Dense(784, 128),
        ActivationLayer(ReLU()),
        Dense(128, 10)
    ]
    x_train = x_train.T
    y_train = y_train.T
    network = Network(layers, softmax=True, loss=mse,
                      loss_prime=mse_prime)
    network.train(x_train, y_train, epochs=1000, alpha=0.05, batch_size=2000)
    predictions = network.prop(x_test)
    accuracy = network.accuracy(predictions, y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Predictions: {np.argmax(predictions, axis=0)}')
    print(f'Actual: {np.argmax(y_test, axis=0)}')


def old_classifier(x_test, y_test, x_train, y_train):
    classifier = Classifier(x_train, y_train, activation=Activation.RELU)
    classifier.train(1000, batch_size=2000, alpha=0.05)
    accuracy, predictions = classifier.test(x_test, y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Predictions: {np.argmax(predictions, axis=0)}')
    print(f'Actual: {np.argmax(y_test, axis=0)}')


def conv_network():
    x_test, y_test, data, labels = load_cifar_data()
    print(data.shape)
    data.reshape((data.shape[1], 3, 32, 32))
    new_data = []
    labels = labels.T
    for i in range(10000):
        col = data[:, i]
        image_r = col[0:1024].reshape((32, 32))
        image_g = col[1024:2048].reshape((32, 32))
        image_b = col[2048:3072].reshape((32, 32))
        image = np.array([image_r, image_g, image_b])
        new_data.append(image)
    new_data = np.asarray(new_data)
    layers = [
        Convolution((32, 32, 3), filters=6, kernel_size=(3, 3)),
        ActivationLayer(ReLU()),
        Reshape((30, 30, 6)),
        Dense(30*30*6, 128),
        ActivationLayer(ReLU()),
        Dense(128, 10),
        ActivationLayer(Sigmoid())
    ]
    network = Network(layers, softmax=False, loss=mse,
                      loss_prime=mse_prime)
    network.train(new_data, labels, epochs=1000, alpha=0.05, batch_size=5000)


def main():
    conv_network()


if __name__ == '__main__':
    main()
