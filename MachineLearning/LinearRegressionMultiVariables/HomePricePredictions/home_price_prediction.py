import pandas as pd
import math
from sklearn import linear_model

dataframe = pd.read_csv("homeprices.csv")

median_bedrooms = math.floor(dataframe.bedrooms.median())
dataframe.bedrooms = dataframe.bedrooms.fillna(median_bedrooms)

rg = linear_model.LinearRegression()
rg.fit(dataframe[['area', 'bedrooms', 'age']], dataframe.price)

print(rg.coef_)
print(rg.intercept_)

print(rg.predict([[3000, 3, 40]]))

