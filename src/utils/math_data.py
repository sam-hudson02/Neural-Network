import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
import pandas as pd
import json
import matplotlib
from utils.utils import one_hot
matplotlib.use('module://matplotlib-backend-kitty')


def load_math_data(path: str, folders: list[str] | None = None):
    # check for folders in path
    if folders is None:
        folders = os.listdir(path)
    meanings = load_math_meta()
    images = []
    labels = []
    for folder in folders:
        print(f'opening folder {folder}')
        i = list(meanings.values()).index(folder)
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

    x, y = process(images, labels)
    x_train = x[:-6000]
    y_train = y[:-6000]
    x_test = x[-6000:]
    y_test = y[-6000:]
    return x_test, y_test, x_train, y_train, meanings


def load_math_meta() -> dict:
    with open('data/math/meta.json', 'r') as f:
        return json.load(f)


def process(images, labels):
    df = pd.DataFrame({'images': images, 'labels': labels})
    # shuffle the data
    df = df.sample(frac=1)
    x = np.array(list(df['images']))
    y = np.array(df['labels'])
    x = x / 255.
    return x, one_hot(y).T


def sample(x, y, meanings):
    for i in range(5):
        key = y[i]
        key = np.argmax(key)
        print(meanings[key])
        plt.imshow(x[i])
        plt.show()
