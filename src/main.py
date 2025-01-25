from utils.data import load_mnist_data
import numpy as np
from utils.utils import Activation, mse, mse_prime
from models.classify import Classifier
from layers.dense import Dense
from layers.activation import Activation as ActivationLayer
from models.nn import Network
from utils.activation import ReLU


def network(x_train, y_train):
    layers = [
        Dense(784, 128),
        ActivationLayer(ReLU()),
        Dense(128, 10)
    ]
    network = Network(layers, softmax=True, loss=mse,
                      loss_prime=mse_prime)
    network.train(x_train, y_train, epochs=1000, alpha=0.05, batch_size=2000)
    return network


def old_classifier(x_test, y_test, x_train, y_train):
    classifier = Classifier(x_train, y_train, activation=Activation.RELU)
    classifier.train(1000, batch_size=2000, alpha=0.05)
    accuracy, predictions = classifier.test(x_test, y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Predictions: {np.argmax(predictions, axis=0)}')
    print(f'Actual: {np.argmax(y_test, axis=0)}')


def main():
    x_test, y_test, x_train, y_train = load_mnist_data()
    nnet = network(x_train, y_train)
    predictions = nnet.prop(x_test)
    accuracy = nnet.accuracy(predictions, y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Predictions: {np.argmax(predictions, axis=0)}')
    print(f'Actual: {np.argmax(y_test, axis=0)}')


if __name__ == '__main__':
    main()
