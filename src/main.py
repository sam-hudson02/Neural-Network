from utils.alpha import exp_alpha, step_alpha
from utils.data import load_cifar_10_data, load_mnist_data, load_cifar_data
import numpy as np
from utils.utils import Activation, mse, mse_prime, one_hot
from models.classify import Classifier
from layers.dense import Dense
from layers.activation import Activation as ActivationLayer
from layers.convolution import Convolution
from layers.reshape import Reshape
from models.nn import Network
from utils.activation import ReLU, Sigmoid, Tanh
import matplotlib
matplotlib.use('module://matplotlib-backend-kitty')
import matplotlib.pyplot as plt


def network():
    x_test, y_test, x_train, y_train = load_mnist_data()
    layers = [
        Dense(784, 128),
        ActivationLayer(ReLU()),
        Dense(128, 10),
        ActivationLayer(Sigmoid())
    ]
    x_train = x_train.T
    y_train = y_train.T
    alpha = exp_alpha(0.5, 0.987)
    network = Network(layers, softmax=True, loss=mse,
                      loss_prime=mse_prime, verbose=True)
    network.train(x_train, y_train, epochs=1000, alpha=alpha, batch_size=2000)
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

def conv_mnist():
    x_test, y_test, x_train, y_train = load_mnist_data()
    print(x_train.shape)
    train_data = []
    for i in range(x_train.shape[1]):
        train_data.append(x_train[:, i].reshape((1, 28, 28)))
    x_train = np.asarray(train_data)
    print(x_train.shape)
    layers = [
        Convolution((28, 28, 1), 10, (3, 3)),
        Reshape((26, 26, 10)),
        Dense(26*26*10, 128),
        ActivationLayer(Tanh()),
        Dense(128, 10),
    ]
    y_train = y_train.T
    network = Network(layers, softmax=True, loss=mse,
                      loss_prime=mse_prime, verbose=True)
    alpha = exp_alpha(0.5, 0.987)
    network.train(x_train, y_train, epochs=200, alpha=alpha, batch_size=4000)


def conv_network():
    x_train, _, y_train, x_test, _, y_test, _ = load_cifar_10_data('data/cifar-10-batches-py')
    print(x_train.shape)
    y_train = one_hot(y_train).T
    x_train = x_train.reshape(x_train.shape[0], 3, 32, 32)
    for i in range(10):
        print(y_train[i])
        plt.imshow(x_train[i].T)
        plt.show()
    layers = [
        Convolution((32, 32, 3), 6, (3, 3)),
        Reshape((30, 30, 6)),
        Dense(30*30*6, 128),
        ActivationLayer(Tanh()),
        Dense(128, 10),
    ]
    network = Network(layers, softmax=True, loss=mse,
                      loss_prime=mse_prime, verbose=True)
    alpha = exp_alpha(0.5, 0.987)
    network.train(x_train, y_train, epochs=1000, alpha=alpha, batch_size=5000)


def main():
    conv_mnist()


if __name__ == '__main__':
    main()
