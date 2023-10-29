import random

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing

df = pd.read_csv("homeprices_bengaluru.csv")

sx = preprocessing.MinMaxScaler()
sy = preprocessing.MinMaxScaler()

scaled_X = sx.fit_transform(df.drop('price', axis='columns'))
scaled_y = sy.fit_transform(df['price'].values.reshape(df.shape[0], 1))


def batch_gradient_descent(X, y_true, epochs, learning_rate=0.01):
    number_of_features = X.shape[1]

    w = np.ones(shape=number_of_features)
    bias = 0
    total_samples = X.shape[0]

    cost_list = []
    epoch_list = []

    for i in range(epochs):
        y_predicted = np.dot(w, scaled_X.T) + bias

        w_grad = -(2 / total_samples) * (X.T.dot(y_true - y_predicted))
        b_grad = -(2 / total_samples) * np.sum(y_true - y_predicted)

        w = w - learning_rate * w_grad
        bias = bias - learning_rate * b_grad

        cost = np.mean(np.square(y_true - y_predicted))

        if i % 10 == 0:
            cost_list.append(cost)
            epoch_list.append(i)

    return w, bias, cost, cost_list, epoch_list


w, bias, cost, cost_list, epoch_list = batch_gradient_descent(scaled_X, scaled_y.reshape(scaled_y.shape[0], ), 500)
print(w, bias, cost)

# plt.xlabel("epoch")
# plt.ylabel("cost")
# plt.plot(epoch_list, cost_list)


# plt.show()

def predict(area, bedrooms, w, b):
    scaled_X = sx.transform([[area, bedrooms]])[0]
    scaled_price = w[0] * scaled_X[0] + w[1] * scaled_X[1] + b
    return sy.inverse_transform([[scaled_price]])[0][0]


print(predict(2600, 4, w, bias))


def stochastic_gradient_descent(X, y_true, epochs, learning_rate=0.01):
    number_of_features = X.shape[1]
    # numpy array with 1 row and columns equal to number of features. In
    # our case number_of_features = 3 (area, bedroom and age)
    w = np.ones(shape=(number_of_features))
    bias = 0
    total_samples = X.shape[0]

    cost_list = []
    epoch_list = []

    for i in range(epochs):

        random_index = random.randint(0, total_samples - 1)

        sample_x = X[random_index]
        sample_y = y_true[random_index]

        y_predicted = np.dot(w, sample_x.T) + bias

        w_grad = -(2 / total_samples) * (sample_x.T.dot(sample_y - y_predicted))
        b_grad = -(2 / total_samples) * np.sum(sample_y - y_predicted)

        w = w - learning_rate * w_grad
        bias = bias - learning_rate * b_grad

        cost = np.mean(np.square(sample_y - y_predicted))

        if i % 100 == 0:
            cost_list.append(cost)
            epoch_list.append(i)

    return w, bias, cost, cost_list, epoch_list


w_sgd, bias_sgd, cost_sgd, cost_list_sgd, epoch_list_sgd = stochastic_gradient_descent(scaled_X, scaled_y.reshape(scaled_y.shape[0], ), 10000)
print(w_sgd, bias_sgd, cost_sgd)

plt.xlabel("epoch")
plt.ylabel("cost")
plt.plot(epoch_list_sgd, cost_list_sgd)
# plt.show()

print(predict(2600, 4, w_sgd, bias_sgd))