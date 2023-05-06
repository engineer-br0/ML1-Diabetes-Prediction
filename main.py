import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score 
from lazypredict.Supervised import LazyClassifier
from xgboost import XGBClassifier

dataset = pd.read_csv("./diabetes.csv")

#print(dataset.head())
#print(dataset.describe())
#print(dataset['Outcome'].value_counts())
# print(dataset.groupby('Outcome').mean())

X = dataset.drop(columns='Outcome')
Y = dataset['Outcome']

scaler = StandardScaler()
scaler.fit(X)
standerized_data = scaler.transform(X)
X = standerized_data
#print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, shuffle=True, test_size=0.15, random_state=100, stratify=Y)
#print(Y_test.head)

# classifier = svm.SVC(kernel='linear')

# classifier.fit(X_train, Y_train)

# X_train_prdiction = classifier.predict(X_train)
# X_test_prediction = classifier.predict(X_test)
# # print("pred", X_train_prdiction)
# # print("test", Y_test)

# X_train_accuracy = accuracy_score(X_train_prdiction, Y_train)
# X_test_accuracy = accuracy_score(X_test_prediction, Y_test)
# # print("X_train_accuracy", X_train_accuracy)
# # print("x_test_accuracy", X_test_accuracy)

# #---------------
# input = (8,183,64,0,0,23.3,0.672,32)
# input_numpy = np.asarray(input)
# input_numpy = input_numpy.reshape(1,-1)
# input = scaler.transform(input_numpy)

# input_prediction = classifier.predict(input)

#print(input_prediction)

clf = LazyClassifier(verbose=0,ignore_warnings=True, custom_metric=None)
classi = clf.fit(X_train, Y_train)

print(models)

