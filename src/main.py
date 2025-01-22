from utils.data import load_mnist_data
import numpy as np
from models.classify import Classifier
from utils.utils import Activation


def main():
    x_test, y_test, x_train, y_train = load_mnist_data()
    print(x_test.shape)
    # save the data to csv
    print(np.argmax(y_test, axis=0))
    classifier = Classifier(x_train, y_train, activation=Activation.RELU)
    classifier.train(200, batch_size=2000)
    accuracy, predictions = classifier.test(x_test, y_test)
    print(f'Accuracy: {accuracy}')
    print(f'Predictions: {np.argmax(predictions, axis=0)}')
    print(f'Actual: {np.argmax(y_test, axis=0)}')


if __name__ == '__main__':
    main()
