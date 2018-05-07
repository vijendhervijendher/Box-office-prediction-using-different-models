import numpy as np
import pandas as pd

# Importing the datasets for training and testing
X = pd.read_csv('trainData.csv')
Y = pd.read_csv('testData.csv')

# Selecting the columns considered as input and output features
X_train = X.iloc[:, :-1].values
X_test = Y.iloc[:, :-1].values

y_train = X.iloc[:, -1].values
y_test = Y.iloc[:, -1].values


from sklearn.metrics import accuracy_score
print ("Accuracy on test data using various algorithms is: \n")
# Logistic Regression
from sklearn.linear_model import LogisticRegression
logRefClassifier = LogisticRegression(random_state = 0, penalty = 'l2', multi_class = 'ovr')
logRefClassifier.fit(X_train, y_train)
y_pred = logRefClassifier.predict(X_test)
print("Logistic Regression for multiclass classification : ", accuracy_score(y_test,y_pred)*100)

# Naive Bayes
from sklearn.naive_bayes import GaussianNB
gnbClassifier = GaussianNB(priors = None)
gnbClassifier.fit(X_train, y_train)
y_pred = gnbClassifier.predict(X_test)
print("Gaussian Naive Bayes : ", accuracy_score(y_test,y_pred)*100)

from sklearn.naive_bayes import MultinomialNB
mnbClassifier = MultinomialNB(priors = None)
mnbClassifier.fit(X_train, y_train)
y_pred = mnbClassifier.predict(X_test)
print("Multinomial Naive Bayes : ", accuracy_score(y_test,y_pred)*100)

# Neural Networks - MultiLayer Perceptron classifier
from sklearn.neural_network import MLPClassifier
mlpClassifier = MLPClassifier(hidden_layer_sizes=(100,), activation='logistic', alpha = 0.001)
mlpClassifier.fit(X_train, y_train)
y_pred = mlpClassifier.predict(X_test)
print("Neural networks - Multilayer Perceptron : ", accuracy_score(y_test,y_pred)*100)

# K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
knnClassifier = KNeighborsClassifier(n_neighbors = 15, metric = 'minkowski', p = 2, algorithm = 'auto', weights = 'uniform')
knnClassifier.fit(X_train, y_train)
y_pred = knnClassifier.predict(X_test)
print("K- nearest neighbours : ", accuracy_score(y_test,y_pred)*100)

# Support Vector Machines
from sklearn.svm import SVC
svcClassifier = SVC(kernel = 'linear',random_state = 0, C = 0.5)
svcClassifier.fit(X_train, y_train)
y_pred = svcClassifier.predict(X_test)
print("Linear SVM : ", accuracy_score(y_test,y_pred)*100)

polySVCClassifier = SVC(kernel = 'poly', random_state = 0, degree = 2, C = 0.5)
polySVCClassifier.fit(X_train, y_train)
y_pred = polySVCClassifier.predict(X_test)
print("Kernel SVM (kernel = polynomial of degree 2) : ", accuracy_score(y_test,y_pred)*100)

rbfSVCclassifier = SVC(kernel = 'rbf', random_state = 0, C = 0.5, decision_function_shape ='ovr')
rbfSVCclassifier.fit(X_train, y_train)
y_pred = rbfSVCclassifier.predict(X_test)
#print("Kernel SVM (kernel = radial basis function) : ", accuracy_score(y_test,y_pred)*100)

# Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
decisionTreeClassifier = DecisionTreeClassifier(criterion = 'gini', random_state = 0, splitter = 'best', max_features = 'auto')
decisionTreeClassifier.fit(X_train, y_train)
y_pred = decisionTreeClassifier.predict(X_test)
print("Decision Tree Classifier : ", accuracy_score(y_test,y_pred)*100)

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
randomForestClassifier = RandomForestClassifier(n_estimators = 40, criterion = 'gini', random_state = 0, max_features = 'auto')
randomForestClassifier.fit(X_train, y_train)
y_pred = randomForestClassifier.predict(X_test)
print("Random Forest Classifier : ", accuracy_score(y_test,y_pred)*100)

# Ensemble Methods
# Adaboost
from sklearn import ensemble
adaBoostClassifier = ensemble.AdaBoostClassifier(n_estimators = 50)
adaBoostClassifier.fit(X_train, y_train)
y_pred = adaBoostClassifier.predict(X_test)
print("AdaBoosting : ", accuracy_score(y_test,y_pred)*100)

# Bagging on various algorithms
from sklearn.ensemble import BaggingClassifier

baggingLRClassifier = BaggingClassifier(LogisticRegression(random_state = 0, penalty = 'l2'), max_samples = 0.4, max_features = 0.5)
baggingLRClassifier.fit(X_train, y_train)
y_pred = baggingLRClassifier.predict(X_test)
#print("Bagging on Logistic Regression : ", accuracy_score(y_test,y_pred)*100)

baggingGNBClassifier = BaggingClassifier(GaussianNB(priors = None), max_samples = 0.4, max_features = 0.4)
baggingGNBClassifier.fit(X_train, y_train)
y_pred = baggingGNBClassifier.predict(X_test)
#print("Bagging on Gaussian Naive Bayes : ", accuracy_score(y_test,y_pred)*100)

baggingKNNClassifier = BaggingClassifier(KNeighborsClassifier(n_neighbors = 15, metric = 'minkowski', p = 2, algorithm = 'auto', weights = 'uniform'), max_samples = 0.5, max_features = 0.5)
baggingKNNClassifier.fit(X_train, y_train)
y_pred = baggingKNNClassifier.predict(X_test)
#print("Bagging on K- nearest neighbours : ", accuracy_score(y_test,y_pred)*100)

baggingSVCClassifier = BaggingClassifier(SVC(kernel = 'linear',random_state = 0, C = 0.5), max_samples = 0.5, max_features = 0.5)
baggingSVCClassifier.fit(X_train, y_train)
y_pred = baggingSVCClassifier.predict(X_test)
print("Bagging on Linear SVM : ", accuracy_score(y_test,y_pred)*100)

baggingPolySVCClassifier = BaggingClassifier(SVC(kernel = 'poly', random_state = 0, degree = 2, C = 0.5),max_samples = 0.5, max_features = 0.5)
baggingPolySVCClassifier.fit(X_train, y_train)
y_pred = baggingPolySVCClassifier.predict(X_test)
#print("Bagging on Kernel SVM (kernel = polynomial of degree 2) : ", accuracy_score(y_test,y_pred)*100)

baggingRBFSVCClassifier = BaggingClassifier(SVC(kernel = 'rbf', random_state = 0, C = 0.5, decision_function_shape ='ovr'),max_samples = 0.5, max_features = 0.5)
baggingRBFSVCClassifier.fit(X_train, y_train)
y_pred = baggingRBFSVCClassifier.predict(X_test)
#print("Bagging on Kernel SVM (kernel = radial basis function) : ", accuracy_score(y_test,y_pred)*100)

baggingDTClassifier = BaggingClassifier(DecisionTreeClassifier(criterion = 'gini', random_state = 0, splitter = 'best', max_features = 'auto'),max_samples = 0.4, max_features = 0.7,random_state = 0)
baggingDTClassifier.fit(X_train, y_train)
y_pred = baggingDTClassifier.predict(X_test)
#print("Bagging on Decision Tree Classifier : ", accuracy_score(y_test,y_pred)*100)

baggingRFClassifier = BaggingClassifier(RandomForestClassifier(n_estimators = 40, criterion = 'gini', random_state = 0, max_features = 'auto'), max_samples = 0.4, max_features = 0.5)
baggingRFClassifier.fit(X_train, y_train)
y_pred = baggingRFClassifier.predict(X_test)
print("Bagging on Random Forest Classifier : ", accuracy_score(y_test,y_pred)*100)
