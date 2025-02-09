import matplotlib.pyplot as plt
from utils.alpha import exp_alpha, step_alpha
from utils.data import load_cifar_10_data, load_sin, load_mnist_data, load_cifar_data
from utils.math_data import load_math_data, load_math_meta
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
from utils.reddit_scrape import load_reddit_data
# matplotlib.use('module://matplotlib-backend-kitty')


def math_classify():
    x_test, y_test, x_train, y_train, meanings = load_math_data(
        'data/math')
    # x_train = x_train[4000:]
    # y_train = y_train[4000:]
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    print(x_train.shape)
    layers = [
        Dense(28 * 28, 128),
        ActivationLayer(Tanh()),
        Dense(128, len(meanings.keys())),
    ]
    network = Network(layers, softmax=True, loss=mse,
                      loss_prime=mse_prime, verbose=True)

    def alpha(i):
        if i < 100:
            return 0.5
        else:
            return 0.3

    network.train(x_train, y_train, epochs=300, alpha=alpha, batch_size=500)
    predictions = network.prop(x_test.T)
    accuracy = network.accuracy(predictions, y_test.T)
    network.save('models/math')
    print(f'Accuracy: {accuracy}')
    print(f'Predictions: {np.argmax(predictions, axis=0)}')
    print(f'Actual: {np.argmax(y_test.T, axis=0)}')
    for i in range(100):
        key = y_test[i]
        key = np.argmax(key)
        pred = np.argmax(predictions.T[i], axis=0)
        print(meanings[key])
        print(meanings[pred])
        plt.imshow(x_test[i].reshape((28, 28)))
        plt.show()


def from_save():
    x_test, y_test, x_train, y_train, meanings = load_math_data(
        'data/test_math')
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    layers = [
        Dense(28 * 28, 128),
        ActivationLayer(Tanh()),
        Dense(128, len(meanings.keys())),
    ]
    network = Network(layers, softmax=True, loss=mse,
                      loss_prime=mse_prime, verbose=True)
    network.open('models/math')
    predictions = network.prop(x_test.T)
    accuracy = network.accuracy(predictions, y_test.T)
    print(f'Accuracy: {accuracy}')
    print(f'Predictions: {np.argmax(predictions, axis=0)}')
    print(f'Actual: {np.argmax(y_test.T, axis=0)}')
    for i in range(10):
        key = y_test[i]
        key = np.argmax(key)
        pred = np.argmax(predictions.T[i], axis=0)
        print(meanings[str(key)])
        print(meanings[str(pred)])
        plt.imshow(x_test[i].reshape((28, 28)))
        plt.show()


def reddit_classify():
    x_test, y_test, x_train, y_train = load_reddit_data()
    layers = [
        Dense(3350, 28*28),
        ActivationLayer(ReLU()),
        Dense(28 * 28, 128),
        ActivationLayer(Tanh()),
        Dense(128, 2),
    ]
    network = Network(layers, softmax=True, loss=mse,
                      loss_prime=mse_prime, verbose=True)

    def alpha(i):
        if i < 100:
            return 0.5
        else:
            return 0.3

    network.train(x_train, y_train, epochs=300, alpha=alpha, batch_size=200)
    predictions = network.prop(x_test.T)
    accuracy = network.accuracy(predictions, y_test.T)
    network.save('models/reddit')
    print(f'Accuracy: {accuracy}')
    print(f'Predictions: {np.argmax(predictions, axis=0)}')
    print(f'Actual: {np.argmax(y_test.T, axis=0)}')


def sin_fit():
    x, y = load_sin()
    layers = [
        Dense(1, 128),
        ActivationLayer(ReLU()),
        Dense(128, 256),
        ActivationLayer(ReLU()),
        Dense(256, 128),
        ActivationLayer(ReLU()),
        Dense(128, 1),
    ]

    def alpha(i):
        if i < 100:
            return 0.00005
        else:
            return 0.00002

    network = Network(layers, softmax=False, loss=mse,
                      loss_prime=mse_prime, verbose=True)

    loss_history = network.train(x, y, epochs=3000, alpha=alpha,
                                 batch_size=200)
    predictions = network.prop(x.T)
    print(f'Predictions: {predictions}')
    print(f'Accuracy: {mse(y, predictions)}')
    # shift x by 2 pi
    plt.plot(x, y)
    plt.plot(x, predictions.T)
    plt.show()
    plt.clf()
    plt.plot(np.log(loss_history))
    plt.show()


def main():
    sin_fit()


if __name__ == '__main__':
    main()
