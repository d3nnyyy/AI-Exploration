import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

df = pd.read_csv("titanic.csv")
df.drop(['PassengerId', 'Name', 'SibSp', 'Parch', 'Ticket', 'Cabin', 'Embarked'], axis='columns', inplace=True)

target = df.Survived
inputs = df.drop('Survived', axis='columns')

dummies = pd.get_dummies(inputs.Sex)

inputs = pd.concat([inputs, dummies], axis='columns')

inputs = df.drop('Sex', axis='columns')

inputs.Age = inputs.Age.fillna(inputs.Age.mean())

print(inputs.head(6))

X_train, X_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2)

model = GaussianNB()

model.fit(X_train, y_train)

print(model.score(X_test, y_test))

print(y_test[:10])
print(model.predict(X_test[:10]))
