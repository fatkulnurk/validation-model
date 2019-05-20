# https://www.pythonforengineers.com/cross-validation-and-model-selection/

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.model_selection import cross_val_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

iris_data = load_iris()
# print(iris_data)


data_input = iris_data.data
data_output = iris_data.target
# print(data_input)
# print(data_output)

kf = KFold(n_splits=5, shuffle=True, random_state=10)
# kf = KFold(n_splits=5)
# print(kf)

print("Train Set          Test Set        ")
for train_set,test_set in kf.split(data_input):
    print(train_set, test_set)

rf_class = RandomForestClassifier(n_estimators=10)
log_class = LogisticRegression()
svm_class = svm.SVC()

# from sklearn import cross_validation
kf2 = KFold(len(iris_data.data), n_folds=10, shuffle=True, random_state=10)
# cv = cross_validation.ShuffleSplit(len(iris_data.data), n_iter=10,  test_size=0.3, random_state=0)
cv = cross_val_score(len(iris_data.data), n_iter=10,  test_size=0.3, random_state=0)