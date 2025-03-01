import matplotlib.pyplot as plt
from utils.alpha import exp_alpha, step_alpha
from utils.data import load_cifar_10_data, load_sin, load_mnist_data, load_cifar_data
from utils.math_data import load_math_data, load_math_meta
import numpy as np
from utils.optimizer import Optimizers
from utils.utils import Activation, cce, cce_softmax_prime, mse, mse_prime, one_hot
from models.classify import Classifier
from layers.dense import Dense
from layers.activation import Activation as ActivationLayer
from layers.convolution import Convolution
from layers.reshape import Reshape
from models.nn import Network
from utils.activation import ReLU, Sigmoid, Tanh
import matplotlib
from utils.reddit_scrape import load_reddit_data
import matplotlib.animation as animation
from plots.plot_sin import sin_gif
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
        ActivationLayer(ReLU()),
        Dense(128, len(meanings.keys())),
    ]
    network = Network(layers, softmax=True, loss=cce,
                      loss_prime=cce_softmax_prime, verbose=True)

    def alpha(i):
        if i < 100:
            return 0.000001
        else:
            return 0.000001

    network.train(x_train, y_train, epochs=300, alpha=alpha, batch_size=500)
    predictions = network.prop(x_test.T)
    accuracy = network.accuracy(predictions, y_test.T)
    network.save('models/math_2')
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


def math_conv():
    x_test, y_test, x_train, y_train, meanings = load_math_data(
        'data/math')
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
    cats = len(meanings.keys())

    filters_1 = 5
    filters_2 = 10
    layers = [
        Convolution((28, 28, 1), filters_1, (3, 3)),
        ActivationLayer(ReLU()),
        Convolution((26, 26, filters_1), filters_2, (3, 3)),
        ActivationLayer(ReLU()),
        Reshape((24, 24, filters_2)),
        Dense(24 * 24 * filters_2, 128),
        ActivationLayer(ReLU()),
        Dense(128, cats),
    ]
    print(x_train.shape)
    network = Network(layers, softmax=True, loss=cce,
                      loss_prime=cce_softmax_prime, verbose=True)

    def alpha(i):
        return 0.000005

    loss = network.train(x_train, y_train, epochs=4,
                         alpha=alpha, batch_size=500)
    network.save('models/math_conv')
    plt.plot(loss)
    plt.show()
    predictions = network.prop(x_test)
    accuracy = network.accuracy(predictions, y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Predictions: {np.argmax(predictions, axis=0)}')
    print(f'Actual: {np.argmax(y_test, axis=0)}')


def conv():
    x_test, y_test, x_train, y_train = load_mnist_data()
    print(x_train.shape)
    print(x_test.shape)
    x_train = x_train.T.reshape(x_train.shape[1], 1, 28, 28)
    x_test = x_test.T.reshape(x_train.shape[1], 1, 28, 28)
    filters_1 = 6
    filters_2 = 12
    opt = Optimizers.ADAM
    layers = [
        Convolution((28, 28, 1), filters_1, (3, 3), opt),
        ActivationLayer(ReLU()),
        Convolution((26, 26, filters_1), filters_2, (3, 3), opt),
        ActivationLayer(ReLU()),
        Reshape((24, 24, filters_2)),
        Dense(24 * 24 * filters_2, 128, opt),
        ActivationLayer(ReLU()),
        Dense(128, 10, opt),
    ]
    print(x_train.shape)
    network = Network(layers, softmax=True, loss=cce,
                      loss_prime=cce_softmax_prime, verbose=True)

    def alpha(i):
        return 0.01

    loss = network.train(x_train, y_train.T, epochs=4,
                         alpha=alpha, batch_size=50)
    plt.plot(loss)
    plt.show()
    predictions = network.prop(x_test.T)
    accuracy = network.accuracy(predictions, y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Predictions: {np.argmax(predictions, axis=0)}')
    print(f'Actual: {np.argmax(y_test, axis=0)}')


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
        ActivationLayer(Sigmoid()),
        Dense(128, 256),
        ActivationLayer(Tanh()),
        Dense(256, 128),
        ActivationLayer(ReLU()),
        Dense(128, 1),
    ]

    def alpha(i):
        if i < 100:
            return 0.00002
        else:
            return 0.00002

    network = Network(layers, softmax=False, loss=mse,
                      loss_prime=mse_prime, verbose=True)

    loss_history = network.train(x, y, epochs=13000, alpha=alpha,
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


def mnist_classify():
    x_test, y_test, x_train, y_train = load_mnist_data()
    print(x_train.shape)
    print(x_test.shape)
    x_train = x_train.T
    opt = Optimizers.ADAM
    layers = [
        Dense(28 * 28, 128, opt),
        ActivationLayer(ReLU()),
        Dense(128, 10, opt),
    ]
    print(x_train.shape)
    network = Network(layers, softmax=True, loss=cce,
                      loss_prime=cce_softmax_prime, verbose=True)

    def alpha(i):
        return 0.2

    loss = network.train(x_train, y_train.T, epochs=10,
                         alpha=alpha, batch_size=64)
    plt.plot(loss)
    plt.show()
    predictions = network.prop(x_test)
    accuracy = network.accuracy(predictions, y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Predictions: {np.argmax(predictions, axis=0)}')
    print(f'Actual: {np.argmax(y_test, axis=0)}')


def main():
    mnist_classify()


if __name__ == '__main__':
    main()
