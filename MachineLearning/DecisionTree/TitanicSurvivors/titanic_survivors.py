import math

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("titanic.csv")
df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)

label_encoder = LabelEncoder()
df['Sex'] = label_encoder.fit_transform(df['Sex'])

inputs = df.drop('Survived', axis='columns')
target = df['Survived']

inputs.Age = inputs.Age.fillna(inputs.Age.mean())

X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)

model = DecisionTreeClassifier()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))
print(model.predict([[2,1,28,18]]))
