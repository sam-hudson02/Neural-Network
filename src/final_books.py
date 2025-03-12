import os
from layers.layer import Layer
from layers.dense import Dense
from layers.dropout import Dropout
from layers.activation import Activation as ActivationLayer
from layers.convolution import Convolution
from layers.reshape import Reshape
from layers.maxpool import MaxPool
from plots.plot_net import plot_compare_acc, plot_compare_loss, plot_multi_accuracy, plot_train_val_acc, plot_train_val_loss
from utils.math_data import load_math_data
from utils.utils import bce, bce_prime, cce, cce_softmax_prime
from models.nn import Network
from utils.activation import ReLU, Sigmoid
from utils.optimizer import Optimizers
import os
from utils.bow import load_data
import numpy as np


def books_fc_multi(epochs: int = 20, sbert: bool = False) -> Network:
    x_train, _, y_train, x_test, _, y_test = load_data(
        sbert=sbert)
    vec_size = x_train.shape[1]
    opt = Optimizers.ADAM
    classes = y_train.shape[1]
    opt = Optimizers.ADAM
    layers: list[Layer] = [
        Dense(vec_size, 128, opt),
        ActivationLayer(ReLU()),
        Dense(128, classes, opt),
        ActivationLayer(Sigmoid()),
    ]
    if sbert:
        name = "all-distilroberta-v1"
    else:
        name = "doc2vec"

    network = Network(layers, softmax=False, loss=bce,
                      name=name,
                      loss_prime=bce_prime, verbose=True)
    vec = 'sbert' if sbert else 'doc2vec'

    mod = f'models/books_fc_128_{epochs}_{vec}_multi'
    if check_model_exists(mod):
        network.open(mod)
    else:
        val = (x_test, y_test)
        network.train(x_train, y_train, epochs=epochs,
                      validation=val,
                      multi=True,
                      batch_size=128)
        network.save(mod)
    predictions = network.prop(x_test.T).T
    pred = predictions[10:].round().astype(int)
    y = y_test[10:]
    print(pred.T)
    print(y.T)
    accuracy = np.mean(np.all(pred == y, axis=1))
    print(f'strict accuracy: {accuracy}')
    return network


def book_conv_multi(epochs: int, sbert: bool = False) -> Network:
    x_train, _, y_train, x_test, _, y_test = load_data(sbert=sbert)
    h = 32
    w = 24
    if not sbert:
        h = 16
        w = 16
    x_train = x_train.reshape(-1, 1, h, w)
    x_test = x_test.reshape(-1, 1, h, w)
    filters_1 = 6
    filters_2 = 12
    l_2_h = (h // 2) - 1
    l_2_w = (w // 2) - 1
    l_3_h = l_2_h - 2
    l_3_w = l_2_w - 2
    opt = Optimizers.ADAM
    classes = y_train.shape[1]
    opt = Optimizers.ADAM
    layers = [
        Convolution((h, w, 1), filters_1, (3, 3), opt),
        ActivationLayer(ReLU()),
        MaxPool(2, 2),
        Convolution((l_2_h, l_2_w, filters_1), filters_2, (3, 3), opt),
        ActivationLayer(ReLU()),
        Reshape((l_3_w, l_3_h, filters_2)),
        Dense(l_3_h * l_3_w * filters_2, 128, opt),
        ActivationLayer(ReLU()),
        Dense(128, classes, opt),
        ActivationLayer(Sigmoid())
    ]
    network = Network(layers, softmax=False, loss=bce,
                      name='CNN with Adam',
                      loss_prime=bce_prime, verbose=True)
    vec = 'sbert' if sbert else 'doc2vec'

    if check_model_exists(f'models/books_conv_128_10_{vec}'):
        network.open(f'models/books_conv_500_10_{vec}')
    else:
        val = (x_test, y_test)
        network.train(x_train, y_train, epochs=epochs,
                      validation=val,
                      multi=True,
                      batch_size=128)
        network.save(f'models/books_conv_128_{epochs}_{vec}')
    predictions = network.prop(x_test.T).T
    pred = predictions[10:].round().astype(int)
    y = y_test[10:]
    print(pred.T)
    print(y.T)
    accuracy = np.mean(np.all(pred == y, axis=1))
    print(f'strict accuracy: {accuracy}')
    return network


def check_model_exists(model_name: str):
    if os.path.exists(model_name):
        return True
    return False


def main():
    fc_doc2vec_multi = books_fc_multi(epochs=10, sbert=False)
    fc_sbert_multi = books_fc_multi(epochs=10, sbert=True)
    _, _, _, x_test, _, y_test = load_data(sbert=True)
    title = 'Class Accuracy Comparison of all-distilroberta-v1'
    file = 'books_sbert_class_acc'
    predictions = fc_sbert_multi.prop(x_test.T).T.round().astype(int)
    plot_multi_accuracy(predictions, y_test, title, file)
    title = 'Validation Loss Comparison'
    file = 'books_fc_loss'
    plot_compare_loss([fc_doc2vec_multi, fc_sbert_multi], title, file)
    title = 'Validation Accuracy Comparison'
    file = 'books_fc_acc'
    plot_compare_acc([fc_doc2vec_multi, fc_sbert_multi], title, file)
    title = 'Training and Validation Loss for Doc2Vec'
    file = 'books_doc2vec_train_val_loss'
    plot_train_val_loss(fc_doc2vec_multi, title, file)
    title = 'Training and Validation Accuracy for Doc2Vec'
    file = 'books_doc2vec_train_val_acc'
    plot_train_val_acc(fc_doc2vec_multi, title, file)
    title = 'Training and Validation Loss for all-distilroberta-v1'
    file = 'books_sbert_train_val_loss'
    plot_train_val_loss(fc_sbert_multi, title, file)
    title = 'Training and Validation Accuracy for all-distilroberta-v1'
    file = 'books_sbert_train_val_acc'
    plot_train_val_acc(fc_sbert_multi, title, file)





if __name__ == '__main__':
    main()
