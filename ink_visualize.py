import numpy as np
import matplotlib.pyplot as plt
import tools


labels, digits = tools.read_data()

# create ink feature
ink = np.array([sum(row) for row in digits])
# compute mean for each digit class
ink_mean = [np.mean(ink[labels == i]) for i in range(10)]
# compute standard deviation for each digit class
ink_std = [np.std(ink[labels == i]) for i in range(10)]
plt.figure(figsize=[10, 10])
x = np.array(range(10))
plt.errorbar(x, ink_mean, yerr=ink_std, fmt='o', ecolor='r', color='b', elinewidth=2, capsize=4)
plt.xlabel('digit classes', fontsize=25, fontweight='semibold')
plt.title('Error bars of ink feature', fontsize=30, fontweight='semibold')
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
plt.savefig('./images/error_bar.pdf')
