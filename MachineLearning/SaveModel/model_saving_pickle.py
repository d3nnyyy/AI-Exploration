import pandas as pd
from sklearn import linear_model
import pickle

dataframe = pd.read_csv("../LinearRegression/HomePricePredictions/home-prices.csv")

model = linear_model.LinearRegression()
model.fit(dataframe[['area']], dataframe.price)

area_to_predict = 3300
print(model.predict([[area_to_predict]]))

with open('model_pickle', 'wb') as f:
    pickle.dump(model, f)

with open('model_pickle', 'rb') as f:
    mp = pickle.load(f)
    print(mp.predict([[area_to_predict]]))