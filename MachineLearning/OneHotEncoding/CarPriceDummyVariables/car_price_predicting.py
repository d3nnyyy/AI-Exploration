import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("car-prices.csv")

dummies = pd.get_dummies(df['Car Model'])

merged = pd.concat([df, dummies], axis='columns')

final = merged.drop(['Car Model', 'Mercedez Benz C class'], axis='columns')

model = LinearRegression()

X = final.drop(['Sell Price($)'], axis='columns')
y = final['Sell Price($)']

print(X)

model.fit(X, y)

print(model.predict([[45000, 4, 0, 0]]))
print(model.predict([[86000, 7, 0, 1]]))

print(model.score(X, y))
