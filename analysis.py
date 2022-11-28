import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tools


def hist(data, save=None, x_label=None, title=None):
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
    plt.tick_params(labelsize=24)
    ax.yaxis.offsetText.set_fontsize(24)
    ax.set_xlabel(x_label, fontsize=25, fontweight='semibold')
    ax.set_ylabel('Count', fontsize=25, fontweight='semibold')
    plt.title(title, fontsize=30, fontweight='semibold')
    fig.show()
    fig.savefig('./images/'+save+'.pdf')


df = pd.read_csv('data/mnist.csv')
values = df.values
print(df.dtypes)
print(df.shape)
print(df.label.value_counts())
print(df.isnull().sum())
print(df.duplicated().sum())

labels, digits = tools.read_data()
no_zero = digits[digits != 0]
hist(digits, save='zero', x_label='pixel values')
hist(no_zero, save='no_zero', x_label='pixel values', title='Histogram of pixel values(non-zero)')
