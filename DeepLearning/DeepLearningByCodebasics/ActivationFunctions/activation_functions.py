import math


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def tanh(x):
    return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))


def relu(x):
    return max(0, x)


def leaky_relu(x):
    return max(0.1 * x, x)


print(sigmoid(100))
print(tanh(100))
print(relu(-2))
print(leaky_relu(-10))
