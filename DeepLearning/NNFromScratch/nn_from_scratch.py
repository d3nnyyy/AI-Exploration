import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("insurance.csv")

X_train, X_test, y_train, y_test = train_test_split(
    df[['age', 'affordability']],
    df.bought_insurance,
    test_size=0.2,
    random_state=25)

X_train_scaled = X_train.copy()
X_train_scaled['age'] = X_train_scaled['age'] / 100

X_test_scaled = X_test.copy()
X_test_scaled['age'] = X_test_scaled['age'] / 100


def log_loss(y_true, y_predicted):
    epsilon = 1e-15
    y_predicted_eps = [max(i, epsilon) for i in y_predicted]
    y_predicted_eps = [min(i, 1 - epsilon) for i in y_predicted_eps]
    y_predicted_eps = np.array(y_predicted_eps)
    return -np.mean(y_true * np.log(y_predicted_eps) + (1 - y_true) * np.log(1 - y_predicted_eps))


def sigmoid_numpy(X):
    return 1 / (1 + np.exp(-X))


class NN:
    def __init__(self):
        self.w1 = 1
        self.w2 = 1
        self.bias = 0

    def fit(self, X, y, epochs, loss_threshold):
        self.w1, self.w2, self.bias = self.gradient_descent(X['age'], X['affordability'], y, epochs, loss_threshold)

    def predict(self, X_test):
        weighted_sum = self.w1 * X_test['age'] + self.w2 * X_test['affordability'] + self.bias
        return sigmoid_numpy(weighted_sum)

    def gradient_descent(self, age, affordability, y_true, epochs, loss_threshold):
        # w1, w2, bias
        w1 = w2 = 1
        bias = 0
        rate = 0.5
        n = len(age)

        for i in range(epochs):
            weighted_sum = w1 * age + w2 * affordability + bias
            y_predicted = sigmoid_numpy(weighted_sum)
            loss = log_loss(y_true, y_predicted)

            w1d = (1 / n) * np.dot(np.transpose(age), (y_predicted - y_true))
            w2d = (1 / n) * np.dot(np.transpose(affordability), (y_predicted - y_true))

            bias_d = np.mean(y_predicted - y_true)

            w1 = w1 - rate * w1d
            w2 = w2 - rate * w2d
            bias = bias - rate * bias_d

            if i % 50 == 0:
                print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')

            if loss < loss_threshold:
                print(f'Epoch:{i}, w1:{w1}, w2:{w2}, bias:{bias}, loss:{loss}')
                break

        return w1, w2, bias


custom_model = NN()
custom_model.fit(X_train_scaled, y_train, epochs=500, loss_threshold=0.4631)
print(custom_model.predict(X_test_scaled))
