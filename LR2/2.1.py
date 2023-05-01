import numpy as np

# Функция активации ReLU
def relu(x):
    return np.maximum(0, x)

# Функция обучения нейронной сети
def train(X, y, num_hidden, learning_rate, num_epochs):
    # Инициализация весов
    input_dim = X.shape[1]
    hidden_weights = np.random.randn(input_dim, num_hidden)
    output_weights = np.random.randn(num_hidden, num_hidden)

    # Цикл обучения
    for epoch in range(num_epochs):
        # Прямой проход
        hidden_layer = relu(X.dot(hidden_weights))
        output_layer = hidden_layer.dot(output_weights)

        # Обратный проход
        output_error = y - output_layer
        hidden_error = output_error.dot(output_weights.T) * (hidden_layer > 0)

        # Обновление весов
        output_weights += hidden_layer.T.dot(output_error) * learning_rate
        hidden_weights += X.T.dot(hidden_error) * learning_rate

    # Возвращаем обученные веса
    return hidden_weights, output_weights

#Тестирование нейронной сети на примере XOR
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

hidden_weights, output_weights = train(X, y, 4, 0.1, 10000)

#Прямой проход на тестовых данных
hidden_layer = relu(X.dot(hidden_weights))
output_layer = hidden_layer.dot(output_weights)
predictions = output_layer.round()

#Вывод результатов
print("Predictions: ", predictions)
print("True labels: ", y)
