import numpy as np
import matplotlib.pyplot as plt
import tools
from concentration import cal_concentration

labels, digits = tools.read_data()
twoD_list = []
avg_list = []

#create concentration_rate feature
for item in digits:
    new_item = np.array(item).reshape([28,28])
    twoD_list.append(new_item)

for item in twoD_list:
    avg_list.append(cal_concentration(item))

con_list = np.array(avg_list)

# compute mean for each digit class
con_mean = [np.mean(con_list[labels == i]) for i in range(10)]
# compute standard deviation for each digit class
con_std = [np.std(con_list[labels == i]) for i in range(10)]

fig = plt.figure(figsize=[10, 10])
x = np.array(range(10))
plt.errorbar(x, con_mean, yerr=con_std, fmt='o', ecolor='r', color='b', elinewidth=2, capsize=4)
plt.xlabel('digit classes', fontsize=25, fontweight='semibold')
plt.title('Error bars of concentration_rate feature', fontsize=30, fontweight='semibold')
plt.xticks(size=20)
plt.yticks(size=20)
plt.show()
fig.savefig('./images/error_bar_concentration.pdf')