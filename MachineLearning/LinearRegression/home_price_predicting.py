import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

dataframe = pd.read_csv("home-prices.csv")

plt.xlabel('area(sqr ft)')
plt.ylabel('area(USD$)')
plt.scatter(dataframe.area, dataframe.price, color='red', marker='+')

rg = linear_model.LinearRegression()
rg.fit(dataframe[['area']], dataframe.price)

area_to_predict = 3300

predicted_price = rg.predict([[area_to_predict]])
print("Predicted price for an area of 3300 sq ft:", predicted_price[0])

plt.show()
