import pandas as pd
from sklearn import linear_model
import joblib

dataframe = pd.read_csv("../LinearRegression/HomePricePredictions/home-prices.csv")

model = linear_model.LinearRegression()
model.fit(dataframe[['area']], dataframe.price)

area_to_predict = 3300
print(model.predict([[area_to_predict]]))

with open('model_joblib', 'wb') as f:
    joblib.dump(model, f)

with open('model_joblib', 'rb') as f:
    mj = joblib.load(f)
    print(mj.predict([[area_to_predict]]))