import pandas as pd
import numpy as np
from sklearn.preprocessing import scale, StandardScaler
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


mnist_data = pd.read_csv('data/mnist.csv').values
labels = mnist_data[:, 0]
digits = mnist_data[:, 1:]
img_size = 28

# create ink feature
ink = np.array([sum(row) for row in digits])
ink = scale(ink).reshape(-1, 1)
# compute mean for each digit class
ink_mean = [np.mean(ink[labels == i]) for i in range(10)]
# compute standard deviation for each digit class
ink_std = [np.std(ink[labels == i]) for i in range(10)]

X_train, X_test, y_train, y_test = train_test_split(ink, labels, test_size=0.2, random_state=42, stratify=labels)
'''lr = LogisticRegression(penalty='l1', solver='saga', multi_class='multinomial')
lr.fit(X_train, y_train)
predict_y = lr.predict(X_test)
print(accuracy_score(y_test, predict_y))'''

clf = LogisticRegressionCV(penalty='l1', solver='saga', multi_class='multinomial', scoring='accuracy')
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))  # score is 0.23
