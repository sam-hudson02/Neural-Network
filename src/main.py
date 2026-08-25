import matplotlib.pyplot as plt
from plots.plot_net import plot_net
from utils.data import load_sin, load_mnist_data
from utils.hasy import generate_index, load_images
from utils.math_data import load_math_data
import numpy as np
from utils.optimizer import Optimizers
from utils.utils import cce, cce_softmax_prime, mse, mse_prime
from layers.dense import Dense
from layers.activation import Activation as ActivationLayer
from layers.convolution import Convolution
from layers.reshape import Reshape
from layers.maxpool import MaxPool
from models.nn import Network
from utils.activation import ReLU, Sigmoid, Tanh
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
    opt = Optimizers.ADAM
    layers = [
        Dense(28 * 28, 128, opt),
        ActivationLayer(ReLU()),
        Dense(128, len(meanings.keys()), opt),
    ]
    network = Network(layers, softmax=True, loss=cce,
                      loss_prime=cce_softmax_prime, verbose=True)

    network.train(x_train, y_train, epochs=10, batch_size=500)
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
        print(meanings[str(key)])
        print(meanings[str(pred)])
        plt.imshow(x_test[i].reshape((28, 28)))
        plt.show()


def from_save():
    x_test, y_test, x_train, y_train, meanings = load_math_data(
        'data/math')
    x_test = x_test.reshape(x_test.shape[0], 28 * 28)
    x_train = x_train.reshape(x_train.shape[0], 28 * 28)
    opt = Optimizers.ADAM
    layers = [
        Dense(28 * 28, 128, opt),
        ActivationLayer(Tanh()),
        Dense(128, len(meanings.keys()), opt),
    ]
    network = Network(layers, softmax=True, loss=cce,
                      loss_prime=cce_softmax_prime, verbose=True)
    network.open('models/math')

    network.train(x_train, y_train, epochs=10, batch_size=500)
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
    x_train = x_train[..., None]
    x_test = x_test[..., None]
    cats = len(meanings.keys())

    opt = Optimizers.ADAM
    filters_1 = 6
    filters_2 = 12
    layers = [
        Convolution((28, 28, 1), filters_1, (3, 3), opt),
        ActivationLayer(ReLU()),
        MaxPool(2, 2),
        Convolution((13, 13, filters_1), filters_2, (3, 3), opt),
        ActivationLayer(ReLU()),
        MaxPool(2, 2),
        Convolution((5, 5, filters_2), filters_2, (3, 3), opt),
        Reshape((3, 3, filters_2)),
        Dense(3 * 3 * filters_2, 128, opt),
        ActivationLayer(ReLU()),
        Dense(128, cats, opt),
    ]
    # layers = [
    #     Convolution((28, 28, 1), filters_1, (3, 3), opt),
    #     ActivationLayer(ReLU()),
    #     Convolution((26, 26, filters_1), filters_2, (3, 3), opt),
    #     ActivationLayer(ReLU()),
    #     Reshape((24, 24, filters_2)),
    #     Dense(24 * 24 * filters_2, 128, opt),
    #     ActivationLayer(ReLU()),
    #     Dense(128, cats, opt),
    # ]
    print(x_train.shape)
    network = Network(layers, softmax=True, loss=cce,
                      loss_prime=cce_softmax_prime, verbose=True)

    val = (x_test, y_test)
    loss = network.train(x_train, y_train, epochs=4,
                         batch_size=512, validation=val)
    network.save('models/math_conv_500')
    loss = network.average_loss()
    acc = network.average_accuracy()
    val_loss = network.validation_loss_history
    val_acc = network.validation_accuracy_history
    plot_net(loss, acc, val_loss, val_acc)
    predictions = network.prop(Network.batch_last(x_test))
    accuracy = network.accuracy(predictions, y_test.T)
    print(f'Accuracy: {accuracy}')
    print(f'Predictions: {np.argmax(predictions, axis=0)}')
    print(f'Actual: {np.argmax(y_test.T, axis=0)}')


def hasy_conv():
    meanings = generate_index('data/hasy/hasy-data-labels.csv')
    x, y = load_images('data/hasy/hasy-data-labels.csv', meanings)
    x = x.reshape(x.shape[0], 28, 28, 1)
    x_train = x[:120000]
    y_train = y[:120000]
    x_test = x[120000:]
    y_test = y[120000:]
    cats = len(meanings.keys())

    opt = Optimizers.ADAM
    filters_1 = 8
    filters_2 = 16
    layers = [
        Convolution((28, 28, 1), filters_1, (3, 3), opt),
        ActivationLayer(ReLU()),
        MaxPool(2, 2),
        Convolution((13, 13, filters_1), filters_2, (3, 3), opt),
        ActivationLayer(ReLU()),
        MaxPool(2, 2),
        Convolution((5, 5, filters_2), filters_2, (3, 3), opt),
        Reshape((3, 3, filters_2)),
        Dense(3 * 3 * filters_2, 128, opt),
        ActivationLayer(ReLU()),
        Dense(128, cats, opt),
    ]
    print(x_train.shape)
    network = Network(layers, softmax=True, loss=cce,
                      loss_prime=cce_softmax_prime, verbose=True)

    loss = network.train(x_train, y_train, epochs=10,
                         batch_size=32)
    network.save('models/hasy_conv')
    plt.plot(loss)
    plt.show()
    predictions = network.prop(Network.batch_last(x_test))
    accuracy = network.accuracy(predictions, y_test.T)
    print(f'Accuracy: {accuracy}')
    print(f'Predictions: {np.argmax(predictions, axis=0)}')
    print(f'Actual: {np.argmax(y_test.T, axis=0)}')


def conv():
    x_test, y_test, x_train, y_train = load_mnist_data()
    print(x_train.shape)
    print(x_test.shape)
    x_train = x_train.T.reshape(-1, 28, 28, 1)
    x_test = x_test.T.reshape(-1, 28, 28, 1)
    y_test = y_test.T
    y_train = y_train.T
    filters_1 = 16
    filters_2 = 32
    opt = Optimizers.ADAM
    layers = [
        Convolution((28, 28, 1), filters_1, (3, 3), opt),
        ActivationLayer(ReLU()),
        MaxPool(2, 2),
        Convolution((13, 13, filters_1), filters_2, (3, 3), opt),
        ActivationLayer(ReLU()),
        MaxPool(2, 2),
        Convolution((5, 5, filters_2), filters_2, (3, 3), opt),
        Reshape((3, 3, filters_2)),
        Dense(3 * 3 * filters_2, 128, opt),
        ActivationLayer(ReLU()),
        Dense(128, 10, opt),
    ]
    print(x_train.shape)
    network = Network(layers, softmax=True, loss=cce,
                      loss_prime=cce_softmax_prime, verbose=True)

    val = (x_test, y_test)

    loss = network.train(x_train, y_train, epochs=10,
                         batch_size=32, validation=val)
    loss = network.average_loss()
    acc = network.average_accuracy()
    val_loss = network.validation_loss_history
    val_acc = network.validation_accuracy_history

    plot_net(loss, acc, val_loss, val_acc)
    predictions = network.prop(Network.batch_last(x_test))
    accuracy = network.accuracy(predictions, y_test.T)
    print(f'Accuracy: {accuracy}')
    print(f'Predictions: {np.argmax(predictions, axis=0)}')
    print(f'Actual: {np.argmax(y_test.T, axis=0)}')


def hasy_classify():
    meanings = generate_index('data/hasy/hasy-data-labels.csv')
    x, y = load_images('data/hasy/hasy-data-labels.csv', meanings)
    x = x.reshape(x.shape[0], 28 * 28)
    x_train = x[:120000]
    y_train = y[:120000]
    x_test = x[120000:]
    y_test = y[120000:]
    cats = len(meanings.keys())
    opt = Optimizers.ADAM
    layers = [
        Dense(28 * 28, 1024, opt),
        ActivationLayer(ReLU()),
        Dense(1024, 512, opt),
        ActivationLayer(ReLU()),
        Dense(512, cats, opt),
    ]
    print(x_train.shape)
    network = Network(layers, softmax=True, loss=cce,
                      loss_prime=cce_softmax_prime, verbose=True)

    predictions = network.prop(x_test.T)
    print(predictions.shape)
    accuracy = network.accuracy(predictions, y_test.T)
    print(f'Accuracy: {accuracy}')
    print(f'Predictions: {np.argmax(predictions, axis=0)}')
    print(f'Actual: {np.argmax(y_test.T, axis=0)}')

    loss = network.train(x_train, y_train, epochs=100,
                         batch_size=2048)
    plt.plot(loss)
    plt.show()
    predictions = network.prop(Network.batch_last(x_test))
    accuracy = network.accuracy(predictions, y_test.T)
    print(f'Accuracy: {accuracy}')
    print(f'Predictions: {np.argmax(predictions, axis=0)}')
    print(f'Actual: {np.argmax(y_test.T, axis=0)}')


def reddit_classify():
    x_test, y_test, x_train, y_train = load_reddit_data()
    layers = [
        Dense(3350, 28*28),
        ActivationLayer(ReLU()),
        Dense(28 * 28, 128),
        ActivationLayer(Tanh()),
        Dense(128, 2),
    ]
    network = Network(layers, softmax=True, loss=cce,
                      loss_prime=cce_softmax_prime, verbose=True)

    network.train(x_train, y_train, epochs=5, batch_size=200)
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

    network = Network(layers, softmax=False, loss=mse,
                      loss_prime=mse_prime, verbose=True)

    loss_history = network.train(x, y, epochs=13000,
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
    # the loader returns one sample per column; train/validate want one per row
    x_train, y_train = x_train.T, y_train.T
    x_test, y_test = x_test.T, y_test.T
    opt = Optimizers.ADAM
    layers = [
        Dense(28 * 28, 128, opt),
        ActivationLayer(ReLU()),
        Dense(128, 10, opt),
    ]
    print(x_train.shape)
    val = (x_test, y_test)
    network = Network(layers, softmax=True, loss=cce,
                      loss_prime=cce_softmax_prime, verbose=True)

    loss = network.train(x_train, y_train, epochs=200,
                         batch_size=32, validation=val)
    loss = network.average_loss()
    acc = network.average_accuracy()
    val_loss = network.validation_loss_history
    val_acc = network.validation_accuracy_history

    plot_net(loss, acc, val_loss, val_acc)
    predictions = network.prop(Network.batch_last(x_test))
    accuracy = network.accuracy(predictions, y_test.T)
    print(f'Accuracy: {accuracy}')
    print(f'Predictions: {np.argmax(predictions, axis=0)}')
    print(f'Actual: {np.argmax(y_test.T, axis=0)}')


def main():
    math_conv()


if __name__ == '__main__':
    main()
