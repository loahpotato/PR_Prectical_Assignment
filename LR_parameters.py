import tools
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

labels, digits = tools.read_data()

X_train, X_test, y_train, y_test = train_test_split(digits, labels, test_size=37000, random_state=42, stratify=labels)

c_range = range(1, 31)
C_error = []

# Cross Verification
for c in c_range:
    lr = LogisticRegression(C=c, penalty='l2', solver='saga',tol=1e-2, multi_class='multinomial', max_iter = 100, n_jobs=-1)
    # 5:1 segment
    scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='accuracy')
    C_error.append(1 - scores.mean())
'''
c = 1
while c <= 4:
    lr = LogisticRegression(C=c, penalty='l1', solver='saga',tol=1e-2, multi_class='multinomial', max_iter = 100, n_jobs=-1)
    scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='accuracy')
    print("Finish c = ", c)
    C_error.append(1 - scores.mean())
    c = c + 0.2
'''
# x axis is c range，y axis is error ratio
plt.plot(c_range, C_error)
plt.xlabel('Value of C for LR')
plt.ylabel('Error')
plt.show()