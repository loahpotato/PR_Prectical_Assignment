import numpy as np
from numpy import reshape
from scipy.spatial import distance
import tools
from sklearn.preprocessing import scale, StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from concentration import cal_concentration

labels, digits = tools.read_data()
twoD_list = []
avg_list = []

for item in digits:
    new_item = np.array(item).reshape([28,28])
    twoD_list.append(new_item)

for item in twoD_list:
    avg_list.append(cal_concentration(item))

#avg_list = np.array(avg_list).reshape(-1, 1)
avg_list = scale(avg_list).reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(avg_list, labels, test_size=0.2, random_state=42, stratify=labels)

clf = LogisticRegressionCV(penalty='l1', solver='saga', multi_class='multinomial', scoring='accuracy')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))  # score is 0.27