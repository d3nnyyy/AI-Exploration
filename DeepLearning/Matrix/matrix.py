import numpy as np

revenue = np.array([[180, 200, 220], [24, 36, 40], [12, 18, 20]])
expenses = np.array([[80, 90, 100], [10, 16, 20], [8, 10, 10]])

profit = revenue - expenses
print(profit)

price_per_unit = np.array([1000, 400, 1200])
units = np.array([[30, 40, 50], [5, 10, 15], [2, 5, 7]])

product = np.dot(price_per_unit, units)
print(product)