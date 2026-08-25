from matplotlib import pyplot as plt
from models.nn import Network
import os
import numpy as np


def plot_net(loss: list[float], acc: list[float], val_loss: list[float],
             val_acc: list[float], title: str = 'Mnist ANN') -> None:
    fig, ax1 = plt.subplots()
    # acc and val_acc on one axis
    ax1.plot(acc, label='Training Accuracy', color='red')
    ax1.plot(val_acc, label='Validation Accuracy', color='green')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    # loss and val_loss on one axis
    ax2 = ax1.twinx()
    ax2.plot(loss, label='Training Loss', color='blue')
    ax2.plot(val_loss, label='Validation Loss', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    plt.savefig(f'./plots/{title}.png')
    plt.close(fig)


def plot_compare_loss(networks: list[Network],
                      title: str = 'Comparison of networks',
                      file: str = 'net') -> None:
    fig, ax = plt.subplots()
    for network in networks:
        ax.plot(network.validation_loss_history, label=network.name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.title(title)
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    plt.savefig(f'./plots/{file}.png')
    plt.close(fig)


def plot_compare_acc(networks: list[Network],
                     title: str = 'Comparison of networks',
                     file: str = 'net') -> None:
    fig, ax = plt.subplots()
    for network in networks:
        ax.plot(network.validation_accuracy_history, label=network.name)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.title(title)
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    plt.savefig(f'./plots/{file}.png')
    plt.close(fig)


def plot_train_val_loss(network: Network,
                        title: str = 'Comparison of networks',
                        file: str = 'net') -> None:
    fig, ax = plt.subplots()
    ax.plot(network.validation_loss_history, label='Validation Loss')
    ax.plot(network.average_loss(), label='Training Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.title(title)
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    plt.savefig(f'./plots/{file}.png')
    plt.close(fig)


def plot_train_val_acc(network: Network,
                       title: str = 'Training and Validation',
                       file: str = 'net') -> None:
    fig, ax = plt.subplots()
    ax.plot(network.validation_accuracy_history, label='Validation Accuracy')
    ax.plot(network.average_accuracy(), label='Training Accuracy')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    plt.title(title)
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    plt.savefig(f'./plots/{file}.png')
    plt.close(fig)


def plot_multi_accuracy(pred: np.ndarray, actual: np.ndarray,
                        title: str = 'Class Accuracy',
                        file: str = 'class_acc') -> None:
    # pred and actual are 2D arrays index, class
    # make bar chart of accuracy for each class
    acc = np.mean(pred == actual, axis=0)
    fig, ax = plt.subplots()
    ax.bar(range(len(acc)), acc)
    ax.set_xlabel('Class')
    ax.set_ylabel('Accuracy')
    plt.title(title)
    if not os.path.exists('./plots'):
        os.makedirs('./plots')
    plt.savefig(f'./plots/{file}.png')
    plt.close(fig)
