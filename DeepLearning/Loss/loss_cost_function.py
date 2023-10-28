import numpy as np

y_predicted = np.array([1, 1, 0, 0, 1])
y_true = np.array([0.30, 0.7, 1, 0, 0.5])


def mae(predicted, true):
    total_error = 0
    for yp, yt in zip(predicted, true):
        total_error += abs(yp - yt)
    print("Total error is:", total_error)
    mae = total_error / len(predicted)
    print("Mean absolute error is:", mae)
    return mae


print("mae using mae function: ", mae(y_predicted, y_true))

print("mae using numpy: ", np.mean(np.abs(y_predicted - y_true)))

epsilon = 1e-15
y_predicted_eps = [max(i, epsilon) for i in y_predicted]
y_predicted_eps = [min(i, 1 - epsilon) for i in y_predicted_eps]
y_predicted_eps = np.array(y_predicted_eps)

print(y_predicted_eps)

print("log loss:", -np.mean(y_true * np.log(y_predicted_eps) + (1 - y_true) * np.log(1 - y_predicted_eps)))
