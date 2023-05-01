import random

# Создание тренировочного набора данных
train_data = []
for i in range(20):
    x1, x2 = random.uniform(0, 1), random.uniform(0, 1)
    if x1 > x2:
        label = 1
    else:
        label = -1
    train_data.append((x1, x2, label))


# Далее реализуем алгоритм перцептрона, который будет обучаться на полученных данных.

# Алгоритм перцептрона
class Perceptron:
    def init(self, num_inputs):
        self.weights = [0] * num_inputs
        self.bias = 0

    def predict(self, inputs):
        activation = 0
        for i in range(len(inputs)):
            activation += inputs[i] * self.weights[i]
        activation += self.bias
        if activation > 0:
            return 1
        else:
            return -1

    def train(self, training_data, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            for x1, x2, label in training_data:
                inputs = [x1, x2]
                prediction = self.predict(inputs)
                if prediction != label:
                    for i in range(len(inputs)):
                        self.weights[i] += learning_rate * label * inputs[i]
                    self.bias += learning_rate * label


# Создание и обучение перцептрона на тренировочном наборе данных
perceptron = Perceptron(num_inputs=2)
perceptron.train(train_data, learning_rate=0.1, num_epochs=100)

# Теперь нужно протестировать точность перцептрона на 1000 случайно сгенерированных точках в единичном квадрате.

# Тестирование перцептрона на 1000 случайных точках
num_correct = 0
for i in range(1000):
    x1, x2 = random.uniform(0, 1), random.uniform(0, 1)
    if x1 > x2:
        true_label = 1
    else:
        true_label = -1
inputs = [x1, x2]
predicted_label = perceptron.predict(inputs)
if predicted_label == true_label:
    num_correct += 1

accuracy = num_correct / 1000
print("Accuracy: ", accuracy)

# Можно увеличить количество точек в тренировочном наборе данных и количество эпох обучения, чтобы улучшить точность модели. Также можно использовать более сложные алгоритмы классификации, если набор данных становится более сложным.
