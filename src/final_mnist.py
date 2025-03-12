from layers.dense import Dense
from layers.activation import Activation as ActivationLayer
from layers.convolution import Convolution
from layers.reshape import Reshape
from layers.maxpool import MaxPool
from plots.plot_net import plot_compare_acc, plot_compare_loss, plot_train_val_acc, plot_train_val_loss
from utils.utils import cce, cce_softmax_prime
from models.nn import Network
from utils.activation import ReLU
from utils.optimizer import Optimizers
from utils.data import load_mnist_data
import os


def mnist_basic(epochs: int = 20):
    print("Running M-NIST Basic with SGD...")
    x_test, y_test, x_train, y_train = load_mnist_data()
    x_train = x_train.T
    # run with gradient descent
    opt = Optimizers.GRAD
    alpha = 0.01
    layers = [
        Dense(28 * 28, 128, opt, alpha),
        ActivationLayer(ReLU()),
        Dense(128, 10, opt, alpha),
    ]
    val = (x_test.T, y_test.T)
    network_1 = Network(layers, softmax=True, loss=cce,
                        loss_prime=cce_softmax_prime, verbose=True,
                        name='FC network with SGD')

    if check_model_exists('models/mnist_basic_sgd'):
        network_1.open('models/mnist_basic_sgd')
    else:
        network_1.train(x_train, y_train.T, epochs=epochs,
                        batch_size=32, validation=val)
        network_1.save('models/mnist_basic_sgd')

    print("Running M-NIST Basic with Adam...")
    # run with adam
    opt = Optimizers.ADAM
    layers = [
        Dense(28 * 28, 128, opt),
        ActivationLayer(ReLU()),
        Dense(128, 10, opt),
    ]
    val = (x_test.T, y_test.T)
    network_2 = Network(layers, softmax=True, loss=cce,
                        loss_prime=cce_softmax_prime, verbose=True,
                        name='FC network with Adam')

    if check_model_exists('models/mnist_basic_adam'):
        network_2.open('models/mnist_basic_adam')
    else:
        network_2.train(x_train, y_train.T, epochs=epochs,
                        batch_size=32, validation=val)
        network_2.save('models/mnist_basic_adam')

    return network_1, network_2


def mnist_conv(epochs: int = 20):
    x_test, y_test, x_train, y_train = load_mnist_data()
    x_train = x_train.T.reshape(x_train.shape[1], 1, 28, 28)
    x_test = x_test.T.reshape(x_test.shape[1], 1, 28, 28)
    y_test = y_test.T
    y_train = y_train.T
    filters_1 = 6
    filters_2 = 12
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
    network = Network(layers, softmax=True, loss=cce,
                      loss_prime=cce_softmax_prime, verbose=True,
                      name='CNN with Adam')

    val = (x_test, y_test)
    if check_model_exists('models/mnist_conv'):
        network.open('models/mnist_conv')
    else:
        network.train(x_train, y_train.T, epochs=epochs,
                      batch_size=32, validation=val)
        network.save('models/mnist_conv')
    return network


def check_model_exists(model_name: str):
    if os.path.exists(model_name):
        return True
    return False


def full():
    fc_sgd, fc_adam = mnist_basic()
    cnn = mnist_conv()
    # plot the results
    title = 'Validation Loss Comparison'
    file = 'mnist_compare_loss'
    plot_compare_loss([fc_sgd, fc_adam, cnn], title, file)
    title = 'Validation Accuracy Comparison'
    file = 'mnist_compare_acc'
    plot_compare_acc([fc_sgd, fc_adam, cnn], title, file)
    title = 'Training and Validation Loss of FC_SGD network'
    file = 'mnist_fc_sgd_loss'
    plot_train_val_loss(fc_sgd, title, file)
    title = 'Training and Validation Loss of FC_Adam network'
    file = 'mnist_fc_adam_loss'
    plot_train_val_loss(fc_adam, title, file)
    title = 'Training and Validation Loss of CNN'
    file = 'mnist_cnn_loss'
    plot_train_val_loss(cnn, title, file)
    title = 'Training and Validation Accuracy of FC_SGD network'
    file = 'mnist_fc_sgd_acc'
    plot_train_val_acc(fc_sgd, title, file)
    title = 'Training and Validation Accuracy of FC_Adam network'
    file = 'mnist_fc_adam_acc'
    plot_train_val_acc(fc_adam, title, file)
    title = 'Training and Validation Accuracy of CNN'
    file = 'mnist_cnn_acc'
    plot_train_val_acc(cnn, title, file)


if __name__ == '__main__':
    full()
