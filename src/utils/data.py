import pandas as pd
import numpy as np
from utils.dtype import DTYPE
from utils.utils import one_hot
from typing import Tuple
import os
from PIL import Image
import pickle

CIFAR_CLASSES = 10


def load_mnist_data(test_size: int = 6000,
                    classes: int | None = None) -> Tuple[np.ndarray, np.ndarray,
                                                         np.ndarray, np.ndarray]:
    """
    Load MNIST as
    x: np.ndarray: The input data, where each column is a sample.
    y: np.ndarray: The output data, where each column is a one-hot
                   encoded label.

    Prefers the official pre-split download, looking for mnist_train.csv and
    mnist_test.csv in data/ and then in data/mnist/. Falls back to the single
    combined data/mnist.csv, holding out the last test_size rows.
    :param test_size: int(optional): Rows held out when splitting the single
                      combined csv. Ignored when the split files exist, which
                      carry their own train/test division.
    :param classes: int(optional): Size of the label space. Defaults to one
                    more than the largest label across both splits, so that
                    EMNIST's larger alphabet works without a code change. The
                    train and test sets always share the same encoding width.
    """
    # the pre-split download, in either of the two places it tends to land
    split_locations = (
        ('./data/mnist_train.csv', './data/mnist_test.csv'),
        ('./data/mnist/mnist_train.csv', './data/mnist/mnist_test.csv'),
    )
    combined = './data/mnist.csv'
    split = next((pair for pair in split_locations
                  if all(os.path.exists(f) for f in pair)), None)

    if split is not None:
        print(f'loading {split[0]} and {split[1]}')
        train_data = np.array(pd.read_csv(split[0]))
        test_data = np.array(pd.read_csv(split[1]))
        np.random.shuffle(train_data)
    elif os.path.exists(combined):
        print(f'loading {combined}, holding out {test_size} rows for testing')
        data = np.array(pd.read_csv(combined))
        np.random.shuffle(data)
        if not 0 < test_size < len(data):
            raise ValueError(f'test_size must be in (0, {len(data)})')
        train_data, test_data = data[:-test_size], data[-test_size:]
    else:
        wanted = ' or '.join(f'{a} and {b}' for a, b in split_locations)
        raise FileNotFoundError(f'need either {wanted}, or {combined}')

    # transpose so that each column is a sample
    train_data = train_data.T
    test_data = test_data.T

    # first row holds the labels, the rest are the pixels
    Y_test = test_data[0].astype(int)
    # divide in the working dtype: 255. is a Python float, so the
    # plain division would hand back float64 pixels
    X_test = test_data[1:].astype(DTYPE) / 255.

    Y_train = train_data[0].astype(int)
    X_train = train_data[1:].astype(DTYPE) / 255.

    # one label space for both splits, or the encodings would not line up
    if classes is None:
        classes = int(max(Y_train.max(), Y_test.max())) + 1

    return np.asarray(X_test), one_hot(Y_test, classes), \
        np.asarray(X_train), one_hot(Y_train, classes)


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_cifar_data():
    """
    Load the CIFAR-10 dataset as (x_test, y_test, x_train, y_train), with one
    sample per column and one-hot labels.
    """
    folder = './data/cifar-10-batches-py/'
    data_train = []
    labels_train = []
    for i in range(1, 6):
        batch = unpickle(folder + f'data_batch_{i}')
        data_train.append(batch[b'data'])
        labels_train.append(np.asarray(batch[b'labels']))
    names = unpickle(folder + 'batches.meta')[b'label_names']
    print(names)
    test_batch = unpickle(folder + 'test_batch')
    data_test = test_batch[b'data'].astype(DTYPE) / 255.
    labels_test = np.asarray(test_batch[b'labels'])

    # stack batches along the sample axis so that row i of data_train still
    # belongs to labels_train[i]; transposing the stacked (5, n, 3072) block
    # instead would interleave the batches and break that pairing
    data_train = np.concatenate(data_train, axis=0).astype(DTYPE) / 255.
    labels_train = np.concatenate(labels_train, axis=0)

    return data_test.T, one_hot(labels_test, CIFAR_CLASSES), \
        data_train.T, one_hot(labels_train, CIFAR_CLASSES)


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
    image_arr = image_arr.astype(DTYPE) / 255.
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
