import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import pandas as pd
import json
from utils.dtype import DTYPE
from utils.utils import one_hot
# matplotlib.use('module://matplotlib-backend-kitty')
from collections import defaultdict


def load_math_data(path: str, folders: list[str] | None = None,
                   test_fraction: float = 0.06,
                   meta_path: str | None = None):
    """
    Load the handwritten maths symbol images under path, one folder per class.
    :param path: str: Dataset root, also where meta.json and count.json live.
    :param folders: list[str](optional): Class folders to read, defaults to
                    every directory under path.
    :param test_fraction: float(optional): Fraction of samples held out for
                          testing, taken after shuffling.
    :param meta_path: str(optional): Where to read meta.json from. Defaults to
                      path, falling back to data/math so that a subset such as
                      data/test_math keeps the full label space.
    """
    # check for folders in path
    if folders is None:
        folders = sorted(f for f in os.listdir(path)
                         if os.path.isdir(os.path.join(path, f)))
    if meta_path is None:
        meta_path = path if os.path.exists(
            os.path.join(path, 'meta.json')) else 'data/math'
    meanings = load_math_meta(meta_path)
    classes = list(meanings.values())
    images = []
    labels = []
    count = defaultdict(int)
    for folder in folders:
        print(f'opening folder {folder}')
        if folder not in classes:
            raise ValueError(f'{folder} is not a class in {path}/meta.json')
        i = classes.index(folder)
        print(i)
        # check for images in folder
        files = os.listdir(os.path.join(path, folder))
        for image in files:
            img = Image.open(os.path.join(path, folder, image))
            img = img.convert('L')
            img = img.resize((28, 28))
            arr = np.array(img)
            images.append(arr)
            labels.append(i)
            count[folder] += 1

    save_count(count, path)

    x, y = process(images, labels, len(classes))
    # scale the split to the dataset so that a small directory still yields
    # both a train and a test set
    n_test = max(1, int(round(len(x) * test_fraction)))
    n_test = min(n_test, len(x) - 1) if len(x) > 1 else len(x)
    x_train, y_train = x[:-n_test], y[:-n_test]
    x_test, y_test = x[-n_test:], y[-n_test:]
    return x_test, y_test, x_train, y_train, meanings


def save_count(count: dict, path: str = 'data/math'):
    with open(os.path.join(path, 'count.json'), 'w') as f:
        json.dump(count, f, indent=2)


def load_math_meta(path: str = 'data/math') -> dict:
    with open(os.path.join(path, 'meta.json'), 'r') as f:
        return json.load(f)


def process(images, labels, classes: int):
    df = pd.DataFrame({'images': images, 'labels': labels})
    # shuffle the data
    df = df.sample(frac=1)
    x = np.array(list(df['images']))
    y = np.array(df['labels'])
    x = x.astype(DTYPE) / 255.
    return x, one_hot(y, classes).T


def sample(x, y, meanings):
    for i in range(5):
        key = y[i]
        key = np.argmax(key)
        # meta.json keys are strings, argmax gives an int
        print(meanings[str(key)])
        plt.imshow(x[i])
        plt.show()
