import pandas as pd
import numpy as np
from utils.utils import one_hot
from typing import Tuple
import os
from PIL import Image


def load_mnist_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray,
                               np.ndarray]:
    """
    Import the classify_data.csv file and return it as
    x: np.ndarray: The input data, where each column is a sample.
    y: np.ndarray: The output data, where each column is a one-hot
                   encoded label.
    """
    data: pd.DataFrame = pd.read_csv('./data/mnist.csv')
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

    return X_test, one_hot(Y_test), X_train, one_hot(Y_train)


def prep_image(image: Image.Image, res: int) -> np.ndarray:
    """
    Preprocess an image for the CNN.
    """
    image = image.resize((res, res))
    image = image.convert('L')
    image = np.array(image)
    image = image.flatten()
    image = image / 255.
    return image


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
