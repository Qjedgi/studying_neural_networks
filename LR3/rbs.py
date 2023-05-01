import matplotlib.pyplot as plt
import numpy as np


def phi(x, c, r):
    return np.exp(-(x - c) ** 2 / (2 * r ** 2)).flatten()


input_dim = 1
hidden_dim = 5
output_dim = 1
centers = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
radius = 1.0
w_hidden = np.zeros((input_dim, hidden_dim))
w_output = np.zeros((hidden_dim, output_dim))
learning_rate = 0.01
num_epochs = 10000
x_train = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]).reshape(-1, 1)
y_train = np.array([-0.48, -0.78, -0.83, -0.67, -0.20, 0.70, 1.48, 1.17, 0.20]).reshape(-1, 1)
for epoch in range(num_epochs):
    hidden_output = np.zeros((x_train.shape[0], hidden_dim))
    for i in range(hidden_dim):
        hidden_output[:, i] = phi(x_train, centers[i], radius)
    output = hidden_output.dot(w_output)
    error = y_train - output
    delta_output = error
    delta_hidden = delta_output.dot(w_output.T) * hidden_output * (1 - hidden_output)
    w_output += learning_rate * hidden_output.T.dot(delta_output)
    w_hidden += learning_rate * x_train.T.dot(delta_hidden)
x_test = np.linspace(-2.0, 2.0, 101).reshape(-1, 1)
hidden_output = np.zeros((x_test.shape[0], hidden_dim))
for i in range(hidden_dim):
    hidden_output[:, i] = phi(x_test, centers[i], radius)
output_test = hidden_output.dot(w_output)
plt.plot(x_train, y_train, 'ro', label='Training data')
plt.plot(x_test, output_test, label='Approximation')
plt.legend()
plt.show()
