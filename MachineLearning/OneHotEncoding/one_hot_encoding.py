import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("homeprices.csv")

dummies = pd.get_dummies(df.town)

merged = pd.concat([df, dummies], axis='columns')

final = merged.drop(['town', 'west windsor'], axis='columns')

model = LinearRegression()

X = final.drop('price', axis='columns')
y = final.price

model.fit(X, y)

# print(model.predict([[2800, 0, 1]]))
# print(model.predict([[3400, 1, 0]]))
# print(model.predict([[4000, 0, 0]]))
#
# print(model.score(X,y))

le = LabelEncoder()

dfle = df

dfle.town = le.fit_transform(dfle.town)

print(dfle)

X = dfle[['town', 'area']].values
y = dfle.price

ohe = OneHotEncoder()

ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])], remainder='passthrough')
X = ct.fit_transform(X)

X = X[:, 1:]

model.fit(X,y)
print(model.predict([[1, 0, 2800]]))