from sklearn import datasets
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB

wine = datasets.load_wine()

df = pd.DataFrame(wine.data, columns=wine.feature_names)

print(df)

df['target'] = wine.target
print(df[50:70])

X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3, random_state=100)

gn = GaussianNB()
gn.fit(X_train, y_train)
print(gn.score(X_test, y_test))

mn = MultinomialNB()
mn.fit(X_train, y_train)
print(mn.score(X_test, y_test))
