import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("homeprices.csv")

dummies = pd.get_dummies(df.town)

merged = pd.concat([df, dummies], axis='columns')

final = merged.drop(['town', 'west windsor'], axis='columns')

model = LinearRegression()

X = final.drop('price', axis='columns')
y = final.price

model.fit(X, y)

print(model.predict([[2800, 0, 1]]))
print(model.predict([[3400, 1, 0]]))
print(model.predict([[4000, 0, 0]]))

print(model.score(X,y))