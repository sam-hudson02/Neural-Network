from layers.dense import Dense
from layers.activation import Activation as ActivationLayer
from layers.convolution import Convolution
from layers.reshape import Reshape
from layers.maxpool import MaxPool
from plots.plot_net import plot_compare_acc, plot_compare_loss, plot_train_val_acc, plot_train_val_loss
from utils.math_data import load_math_data
from utils.utils import cce, cce_softmax_prime
from models.nn import Network
from utils.activation import ReLU
from utils.optimizer import Optimizers
import os


def math_fc(epochs: int = 10):
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
    val = (x_test, y_test)
    network = Network(layers, softmax=True, loss=cce,
                      name='FC Network with Adam',
                      loss_prime=cce_softmax_prime, verbose=True)

    if check_model_exists('models/math_fc_500_10'):
        network.open('models/math_fc_500_10')
    else:
        network.train(x_train, y_train, epochs=epochs,
                      validation=val,
                      batch_size=512)
        network.save('models/math_fc_500_10')
    return network


def math_conv(epochs: int = 10):
    x_test, y_test, x_train, y_train, meanings = load_math_data(
        'data/math')
    x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
    x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
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
                      name='CNN with Adam',
                      loss_prime=cce_softmax_prime, verbose=True)

    val = (x_test, y_test)
    if check_model_exists('models/math_conv_500_10'):
        network.open('models/math_conv_500_10')
    else:
        network.train(x_train, y_train, epochs=epochs,
                      batch_size=512, validation=val)
        network.save('models/math_conv_500_10')
    return network


def check_model_exists(model_name: str):
    if os.path.exists(model_name):
        return True
    return False


def main():
    fc = math_fc(10)
    conv = math_conv(10)
    file_name = 'math_loss_compare_500_10'
    title = 'Validation Loss Comparison'
    plot_compare_loss([fc, conv], title, file_name)
    file_name = 'math_acc_compare_500_10'
    title = 'Validation Accuracy Comparison'
    plot_compare_acc([fc, conv], title, file_name)
    file_name = 'math_loss_fc_500_10'
    title = 'Training and Validation Loss for FC'
    plot_train_val_loss(fc, title, file_name)
    file_name = 'math_acc_fc_500_10'
    title = 'Training and Validation Accuracy for FC'
    plot_train_val_acc(fc, title, file_name)
    file_name = 'math_loss_conv_500_10'
    title = 'Training and Validation Loss for Conv'
    plot_train_val_loss(conv, title, file_name)
    file_name = 'math_acc_conv_500_10'
    title = 'Training and Validation Accuracy for Conv'
    plot_train_val_acc(conv, title, file_name)


if __name__ == '__main__':
    main()
