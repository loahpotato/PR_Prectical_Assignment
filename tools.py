import pandas as pd
import matplotlib.pyplot as plt


def read_data():
    mnist_data = pd.read_csv('data/mnist.csv').values
    labels = mnist_data[:, 0]
    digits = mnist_data[:, 1:]
    # img_size = 28
    return labels, digits  # , img_size


def view_digit(index=0, img_size=28):
    labels, digits = read_data()
    fig = plt.figure()
    plt.imshow(digits[index].reshape(img_size, img_size))
    plt.show()
    fig.savefig('./images/view.pdf')


# view_digit()

