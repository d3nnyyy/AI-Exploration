import pandas as pd
from sklearn import linear_model

dataframe = pd.read_csv("marks.csv")

print(dataframe)

rg = linear_model.LinearRegression()
rg.fit(dataframe[['mark_by_coworkers', 'mark_by_teacher', 'coefficient', 'team_mark']], dataframe.final_mark)

print(rg.predict([[20, 20, 1, 50]]))
