# Tutorial https://www.ritchieng.com/machine-learning-cross-validation/

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
# from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# read in the iris data
iris = load_iris()

# create X (features) and y (response)
X = iris.data
y = iris.target

# use train/test split with different random_state values
# we can change the random_state values that changes the accuracy scores
# the accuracy changes a lot
# this is why testing accuracy is a high-variance estimate
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=6)

# check classification accuracy of KNN with K=5
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
result_metrics_accuracy_score = metrics.accuracy_score(y_test, y_pred)


# simulate splitting a dataset of 25 observations into 5 folds
# from sklearn.model_selection import KFold
# kf = KFold(random_state=25, n_splits=5, shuffle=False)

# for Loo
from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.get_n_splits(X)

# print the contents of each training and testing set
# ^ - forces the field to be centered within the available space
# .format() - formats the string similar to %s or %n
# enumerate(sequence, start=0) - returns an enumerate object
print('{} {:^61} {}'.format('Iteration', 'Training set obsevations', 'Testing set observations'))
for iteration, data in enumerate(loo.split(X), start=1):
    print('{!s:^9} {} {!s:^25}'.format(iteration, data[0], data[1]))


from sklearn.model_selection import cross_val_score
# 10-fold cross-validation with K=5 for KNN (the n_neighbors parameter)
# k = 5 for KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)

# Use cross_val_score function
# We are passing the entirety of X and y, not X_train or y_train, it takes care of splitting the dat
# cv=10 for 10 folds
# scoring='accuracy' for evaluation metric - althought they are many
scores = cross_val_score(knn, X, y, cv=10, scoring='accuracy')

print('Score')
# print(scores)
print('------------------------------------------')
print('Iterasi\t| Score')
for i in range (len(scores)):
    print(i+1, '\t|', scores[i])
print('------------------------------------------')
# use average accuracy as an estimate of out-of-sample accuracy
# numpy array has a method mean()
print('Rata Rata', scores.mean())