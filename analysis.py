import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def hist(data, save=None, x_label=None):
    b = [i for i in range(0, 261, 10)]
    hists, bins = np.histogram(data, bins=b)
    fig = plt.figure(figsize=[15, 15])
    ax = fig.subplots()
    n, bins, patches = ax.hist(bins[:-1], bins, weights=hists, rwidth=0.9)

    bins = bins[1:]
    col = (bins - bins.min()) / (bins.max() - bins.min())
    cm = plt.cm.get_cmap('viridis')
    for c, p in zip(col, patches):
        plt.setp(p, 'facecolor', cm(c))

    plt.tick_params(labelsize=20)
    ax.set_xlabel(x_label, fontsize=20, fontweight='semibold')
    ax.set_ylabel('Count', fontsize=20, fontweight='semibold')
    fig.show()
    # fig.savefig(save+'.pdf')


'''df = pd.read_csv('data/mnist.csv')
df1 = df.iloc[0]
print(df1)
print(df1.describe())'''

mnist_data = pd.read_csv('data/mnist.csv').values
labels = mnist_data[:, 0]
digits = mnist_data[:, 1:]
img_size = 28
# print(np.max(digits))
mask = digits != 0
new_data = digits[mask]
hist(new_data)
