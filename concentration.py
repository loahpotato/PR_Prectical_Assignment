import numpy as np
from numpy import reshape
from scipy.spatial import distance
import tools
from sklearn.preprocessing import scale, StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

labels, digits = tools.read_data()

# calculate concentration_rate feature
def cal_concentration(image):

    black_point_list = []
    sum_x = 0
    sum_y = 0
    dis_list = []

    for num_x, rows in enumerate(image):
        for num_y, column in enumerate(rows):
            if column > 200:
                black_point_list.append([num_x, num_y])
    
    for item in black_point_list:
        sum_x += item[0]
        sum_y += item[1]

    val_x = sum_x / len(black_point_list)
    val_y = sum_y / len(black_point_list)

    for item in black_point_list:
        dis_list.append( distance.euclidean( [val_x, val_y], item ) )
    avg = sum(dis_list) / len(dis_list)

    return avg

twoD_list = []
avg_list = []

for item in digits:
    new_item = np.array(item).reshape([28,28])
    twoD_list.append(new_item)

for item in twoD_list:
    avg_list.append(cal_concentration(item))

avg_list = np.array(avg_list).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(avg_list, labels, test_size=0.2, random_state=42, stratify=labels)

clf = LogisticRegressionCV(penalty='l1', solver='saga', multi_class='multinomial', scoring='accuracy')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))  # score is 0.27
