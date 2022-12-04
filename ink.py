import numpy as np
import tools
from sklearn.preprocessing import scale, StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

labels, digits = tools.read_data()

# create ink feature
ink = np.array([sum(row) for row in digits])
ink = scale(ink).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(ink, labels, test_size=0.2, random_state=42, stratify=labels)
'''lr = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial')
lr.fit(X_train, y_train)
predict_y = lr.predict(X_test)
print(accuracy_score(y_test, predict_y))'''

clf = LogisticRegressionCV(penalty='l1', solver='saga', multi_class='multinomial', scoring='accuracy')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))  # score is 0.23


clf.fit(ink, labels)
predict_y = clf.predict(X_test)

# calculate confusion matrix and plot
matrix = confusion_matrix(y_test, clf.predict(X_test))

accu = []
for num,item in enumerate(matrix):
    accu.append( item[num] / sum(item) )
print(accu)

ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.show()