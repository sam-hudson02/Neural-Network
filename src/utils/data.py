import pandas as pd
import numpy as np
from utils.utils import one_hot
from typing import Tuple
import os
from PIL import Image
import pickle
import matplotlib.pyplot as plt


def load_mnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray]:
    """
    Import the classify_data.csv file and return it as
    x: np.ndarray: The input data, where each column is a sample.
    y: np.ndarray: The output data, where each column is a one-hot
                   encoded label.
    """
    data = pd.read_csv('./data/mnist.csv')
    data = np.array(data)
    np.random.shuffle(data)

    # transpose the data
    data = data.T

    rows = data.shape[0]

    # take the first 1000 cols as the test data
    data_test = data[:, 0:1000]
    # gets the first rows as the labels
    Y_test = data_test[0]
    # gets the rest of the data as the input
    X_test = data_test[1:rows]
    # normalize the data
    X_test = X_test / 255.

    # take the rest of the cols as the training data
    data_train = data[:, 1000:]
    # gets the first rows as the labels
    Y_train = data_train[0]
    # gets the rest of the data as the input
    X_train = data_train[1:rows]
    # normalize the data
    X_train = X_train / 255

    return np.asarray(X_test), one_hot(Y_test), \
        np.asarray(X_train), one_hot(Y_train)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar_data():
    """
    Load the CIFAR-10 dataset.
    """
    folder = './data/cifar-10-batches-py/'
    data_train = []
    labels_train = []
    for i in range(1, 6):
        batch = unpickle(folder + f'data_batch_{i}')
        data_train.append(batch[b'data'])
        labels_train.append(batch[b'labels'])
    names = unpickle(folder + 'batches.meta')[b'label_names']
    print(names)
    test_batch = unpickle(folder + 'test_batch')
    data_test = test_batch[b'data']
    labels_test = np.asarray(test_batch[b'labels'])
    labels_train = np.array(labels_train)
    data_train = np.asarray(data_train).T.reshape((3072, -1))
    data_train = data_train / 255.
    return data_test.T, one_hot(labels_test), \
        data_train, one_hot(labels_train)


def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072

    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)

    # training data
    cifar_train_data = None
    cifar_train_filenames = []
    cifar_train_labels = []

    # cifar_train_data_dict
    # 'batch_label': 'training batch 5 of 5'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack(
                (cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_filenames += cifar_train_data_dict[b'filenames']
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape(
        (len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(
            0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_labels = np.array(cifar_train_labels)

    # test data
    # cifar_test_data_dict
    # 'batch_label': 'testing batch 1 of 1'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_filenames = cifar_test_data_dict[b'filenames']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape(
        (len(cifar_test_data), 32, 32, 3))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(
            0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_labels = np.array(cifar_test_labels)

    return cifar_train_data, cifar_train_filenames, cifar_train_labels, \
        cifar_test_data, cifar_test_filenames, cifar_test_labels, cifar_label_names


def prep_image(image: Image.Image, res: int) -> np.ndarray:
    """
    Preprocess an image for the CNN.
    """
    image = image.resize((res, res))
    image = image.convert('L')
    image_arr = np.array(image)
    image_arr = image_arr.flatten()
    image_arr = image_arr / 255.
    return image_arr


def covid_image_data(res: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    folder = './data/covid_2/'
    covid_folder = os.path.join(folder, 'covid')
    normal_folder = os.path.join(folder, 'normal')
    pneumonia_folder = os.path.join(folder, 'pneumonia')
    images = []
    vectors = []
    for filename in os.listdir(covid_folder):
        img = Image.open(os.path.join(covid_folder, filename))
        img = prep_image(img, res)
        images.append(img)
        vectors.append(np.array([1, 0, 0]))
    for filename in os.listdir(normal_folder):
        img = Image.open(os.path.join(normal_folder, filename))
        img = prep_image(img, res)
        images.append(img)
        vectors.append(np.array([0, 1, 0]))
    for filename in os.listdir(pneumonia_folder):
        img = Image.open(os.path.join(pneumonia_folder, filename))
        img = prep_image(img, res)
        images.append(img)
        vectors.append(np.array([0, 0, 1]))
    # create df with vectors and images
    images = np.array(images)
    vectors = np.array(vectors)
    print(images.shape)
    print(vectors.shape)
    # connect data based on index and shuffle
    data = np.column_stack((vectors, images))
    np.random.shuffle(data)
    x = data[:, 3:].T
    y = data[:, :3].T
    print(x.shape)
    print(y.shape)
    return x, y


def load_sin(data_points: int = 2000,
             noise: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the sin dataset.
    """
    x = np.linspace(0, 2 * np.pi, data_points).reshape(data_points, 1)
    y = np.sin(x).reshape(data_points, 1)
    if noise:
        y += np.random.normal(0, 0.1, data_points).reshape(data_points, 1)
    return x, y
