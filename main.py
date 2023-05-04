import pandas as pd
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("./diabetes.csv")

#print(dataset.head())
#print(dataset.describe())
#print(dataset['Outcome'].value_counts())
# print(dataset.groupby('Outcome').mean())

X = dataset.drop(columns='Outcome')
Y = dataset['Outcome']
standerized_data = StandardScaler().fit_transform(X)
X = standerized_data
print(X)

