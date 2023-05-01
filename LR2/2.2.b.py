import random

def adaline_train(data, learning_rate=0.1, epochs=100):
    # Инициализация весов и порога случайными значениями
    w1 = random.uniform(-1, 1)
    w2 = random.uniform(-1, 1)
    t = random.uniform(-1, 1)

    for epoch in range(epochs):
        # Обход всех точек в наборе данных
        for x1, x2, label in data:
            # Вычисление выходного значения нейрона
            net = x1 * w1 + x2 * w2 - t
            # Обновление весов и порога в соответствии с ошибкой
            w1 += learning_rate * (label - net) * x1
            w2 += learning_rate * (label - net) * x2
            t -= learning_rate * (label - net)

    return w1, w2, t

# Создание тренировочного набора данных
train_data = []
for i in range(20):
    x1 = random.uniform(0, 1)
    x2 = random.uniform(0, 1)
    if x1 > x2:
        label = 1
    else:
        label = -1
    train_data.append((x1, x2, label))

# Обучение нейрона типа адалайн на тренировочном наборе данных
w1, w2, t = adaline_train(train_data)

# Создание набора тестовых данных и проверка точности классификации
test_data = []
for i in range(20):
    x1 = random.uniform(0, 1)
    x2 = random.uniform(0, 1)
    if x1 > x2:
        label = 1
    else:
        label = -1
    test_data.append((x1, x2, label))

correct = 0
for x1, x2, label in test_data:
    net = x1 * w1 + x2 * w2 - t
    if (net >= 0 and label == 1) or (net < 0 and label == -1):
        correct += 1

accuracy = correct / len(test_data)
print("Accuracy:", accuracy)
