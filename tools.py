import pandas as pd


def read_data():
    mnist_data = pd.read_csv('data/mnist.csv').values
    labels = mnist_data[:, 0]
    digits = mnist_data[:, 1:]
    # img_size = 28
    return labels, digits  # , img_size

