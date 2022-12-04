import numpy as np
from numpy import reshape
from scipy.spatial import distance
import tools
from sklearn.preprocessing import scale, StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from concentration import cal_concentration
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

labels, digits = tools.read_data()
twoD_list = []
avg_list = []

# concentration_rate 
for item in digits:
    new_item = np.array(item).reshape([28,28])
    twoD_list.append(new_item)

for item in twoD_list:
    avg_list.append(cal_concentration(item))

avg_list = scale(avg_list)

# ink
ink = [sum(row) for row in digits]
ink = scale(ink)

combined_list = np.array( [[avg_list[i], ink[i]] for i in range(len(digits))] )
print(combined_list)

X_train, X_test, y_train, y_test = train_test_split(combined_list, labels, test_size=0.2, random_state=42, stratify=labels)

clf = LogisticRegressionCV(penalty='l1', solver='saga', multi_class='multinomial', scoring='accuracy')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))  # score is 0.33


# calculate confusion matrix and plot
matrix = confusion_matrix(y_test, clf.predict(X_test))

accu = []
for num,item in enumerate(matrix):
    accu.append( item[num] / sum(item) )
print(accu)

ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.show()