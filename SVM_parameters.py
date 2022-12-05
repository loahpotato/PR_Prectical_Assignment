import tools
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

labels, digits = tools.read_data()

X_train, X_test, y_train, y_test = train_test_split(digits, labels, test_size=37000, random_state=42, stratify=labels)

c_range = range(1, 31)
C_error = []
# Cross Verification
for c in c_range:
    clf = svm.SVC(C=c, tol=1e-3)
    # 5:1 segment
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
    C_error.append(1 - scores.mean())

# x axis is c range，y axis is error ratio
plt.plot(c_range, C_error)
plt.xlabel('Value of C for SVM')
plt.ylabel('Error')
plt.show()