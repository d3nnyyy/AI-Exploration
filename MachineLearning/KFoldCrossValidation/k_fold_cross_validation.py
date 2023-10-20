from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np

digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.3)

lr = LogisticRegression()
lr.fit(X_train, y_train)
print(lr.score(X_test, y_test))

svc = SVC()
svc.fit(X_train, y_train)
print(svc.score(X_test, y_test))

rfc = RandomForestRegressor()
rfc.fit(X_train, y_train)
print(rfc.score(X_test, y_test))

kf = KFold(n_splits=3)
for train_index, test_index in kf.split([1, 2, 3, 4, 5, 6, 7, 8, 9]):
    print(train_index, test_index)


def get_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)


print(get_score(LogisticRegression(), X_train, X_test, y_train, y_test))

folds = StratifiedKFold(n_splits=3)

scores_l = []
scores_svc = []
scores_rfc = []

for train_index, test_index in kf.split(digits.data):
    X_train, X_test, y_train, y_test = \
        digits.data[train_index], digits.data[test_index], digits.target[train_index], digits.target[test_index]

    scores_l.append(get_score(LogisticRegression(), X_train, X_test, y_train, y_test))
    scores_svc.append(get_score(SVC(), X_train, X_test, y_train, y_test))
    scores_rfc.append(get_score(RandomForestRegressor(), X_train, X_test, y_train, y_test))

print(scores_l, scores_rfc, scores_svc)

print(cross_val_score(LogisticRegression(), digits.data, digits.target))