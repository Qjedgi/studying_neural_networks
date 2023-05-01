import numpy as np


class HammingNeuralNetwork:
    def __init__(self, p, N, theta):
        self.p = p
        self.N = N
        self.theta = theta
        self.W = np.zeros((self.N, self.N))

    def train(self, X):
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    self.W[i][j] += np.sum(X[:, i] * X[:, j])
        self.W[self.W < self.theta] = 0

    def recognize(self, X):
        distances = np.zeros((self.p,))
        for i in range(self.p):
            distances[i] = np.sum(np.abs(self.W[i % 7] - X[0]))
        return np.argmax(distances)


p = 10
N = 7
reference_samples = np.array([
    [-1, 1, 1, 1, -1, -1, -1],
    [-1, -1, -1, 1, -1, -1, -1],
    [-1, 1, -1, 1, -1, 1, -1],
    [1, 1, 1, 1, -1, -1, -1],
    [1, -1, -1, 1, -1, -1, -1],
    [-1, 1, -1, -1, -1, 1, -1],
    [-1, 1, 1, 1, 1, 1, -1],
    [1, -1, -1, -1, -1, -1, 1],
    [-1, 1, -1, 1, -1, 1, -1],
    [-1, 1, 1, 1, -1, 1, -1]
])
test_images = np.array([
    [-1, 1, 1, 1, 1, -1, -1],
    [-1, -1, -1, 1, -1, -1, 1],
    [-1, 1, -1, 1, 1, 1, 1],
    [1, 1, -1, 1, -1, -1, -1],
    [1, -1, 1, 1, 1, -1, -1],
    [-1, 1, -1, -1, -1, 1, -1],
    [-1, 1, 1, 1, 1, 1, 1],
    [1, -1, -1, -1, -1, -1, 1],
    [-1, 1, -1, 1, -1, 1, -1],
    [-1, 1, 1, 1, -1, 1, -1]
])
test_images_with_noise = np.copy(test_images)
for i in range(test_images_with_noise.shape[0]):
    for j in range(test_images_with_noise.shape[1]):
        if np.random.uniform(0, 1) < 0.1:
            test_images_with_noise[i][j] *= -1
hamming_nn = HammingNeuralNetwork(p, N, 0)
hamming_nn.train(reference_samples)
correct_count: int = 0
for i in range(test_images_with_noise.shape[0]):
    recognized_digit = hamming_nn.recognize(test_images_with_noise[i].reshape(1, -1))
    if recognized_digit - 1 == i:
        correct_count += 1
    print(f"Test image {i + 1} recognized as {recognized_digit}")
print(f"Number of correct answers is {correct_count}")
