import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

df = pd.read_csv("car-prices.csv")

plt.scatter(df['Mileage'], df['Sell Price($)'])
plt.show()

X = df[['Mileage']]
y = df[['Sell Price($)']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)

print(clf.predict(X_test))
print(y_test)
print(clf.score(X_test, y_test))