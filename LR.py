import numpy as np
import tools
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

labels, digits = tools.read_data()
digits = np.array(digits)
X_train, X_test, y_train, y_test = train_test_split(digits, labels, test_size=37000, random_state=42, stratify=labels)
print(len(X_train))
clf = LogisticRegression(penalty='l1', solver='saga', C=1 , tol=1e-2, multi_class='multinomial', max_iter = 100, n_jobs=-1)
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))  # score is 0.89

# calculate confusion matrix and accuracy then plot
t = classification_report(y_test, clf.predict(X_test), labels=[0,1,2,3,4,5,6,7,8,9])
print(t)
ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
plt.show()