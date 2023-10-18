import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

dataframe = pd.read_csv("home-prices.csv")

rg = linear_model.LinearRegression()
rg.fit(dataframe[['area']], dataframe.price)

area_to_predict = 3300

predicted_price = rg.predict([[area_to_predict]])
print(predicted_price[0])

d = pd.read_csv("areas.csv")
p = rg.predict(d)

d['prices'] = p
print(d)

d.to_csv('prediction.csv', index=False)

plt.xlabel('area(sqr ft)')
plt.ylabel('area(USD$)')
plt.scatter(dataframe.area, dataframe.price, color='red', marker='+')
plt.plot(dataframe.area, rg.predict(dataframe[['area']]), color='blue')

plt.show()
