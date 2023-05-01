import numpy as np
from numpy import ndarray


class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.random.normal(size=(size, size))
        np.fill_diagonal(self.weights, 0)
        self.bias = np.sum(self.weights, axis=1)

    def train(self, patterns_matrix):
        num_patterns = patterns_matrix.shape[0]
        for p in range(num_patterns):
            pattern = patterns_matrix[p]
            self.weights += np.outer(pattern - self.bias, pattern - self.bias)
        np.fill_diagonal(self.weights, 0)
        self.bias = np.sum(self.weights, axis=1)

    def test(self, input_pattern, patterns_matrix, max_iterations=100):
        state = np.copy(input_pattern)
        for i in range(max_iterations):
            prev_state = np.copy(state)
            -0.5 * np.dot(np.dot(state - self.bias, self.weights), state - self.bias)
            state = np.heaviside(np.dot(self.weights, state) - self.bias, 0)
            if np.max(np.abs(state - prev_state)) < 1e-6:
                break
        best_pattern: None = None
        min_distance = float('inf')
        for pattern in patterns_matrix:
            distance = np.sum(np.abs(pattern - state))
            if distance < min_distance:
                min_distance = distance
                best_pattern = pattern
        return best_pattern


image1: ndarray = np.array([[1, -1, -1, -1, -1, -1, -1, -1, -1, 1],
                            [1, 1, 1, 1, 1, -1, -1, 1, -1, -1],
                            [1, -1, 1, 1, -1, -1, -1, -1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                            [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                            [-1, -1, -1, 1, -1, -1, -1, 1, -1, 1],
                            [1, -1, 1, 1, -1, -1, 1, -1, 1, 1],
                            [1, 1, -1, -1, -1, 1, -1, 1, 1, 1],
                            [1, 1, -1, -1, 1, -1, -1, 1, 1, 1],
                            [-1, -1, 1, -1, -1, 1, -1, -1, 1, -1]])
image2: ndarray = np.array([[1, 1, -1, -1, -1, -1, 1, -1, -1, 1],
                            [1, -1, 1, 1, 1, -1, -1, 1, 1, -1],
                            [1, -1, 1, 1, -1, -1, -1, -1, -1, 1],
                            [1, 1, 1, -1, 1, -1, 1, 1, 1, 1],
                            [-1, -1, -1, -1, 1, -1, 1, 1, -1, -1],
                            [-1, -1, -1, 1, 1, 1, -1, 1, 1, 1],
                            [1, -1, 1, 1, -1, -1, 1, -1, -1, 1],
                            [1, -1, 1, -1, 1, 1, -1, 1, 1, 1],
                            [1, 1, 1, 1, 1, -1, -1, 1, 1, 1],
                            [-1, -1, 1, -1, -1, 1, 1, -1, 1, -1]])
test_data = np.array([[-1, -1, 1, -1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, 1, 1, -1, -1, -1, -1, -1, 1],
                      [1, 1, 1, -1, -1, -1, -1, -1, -1, 1],
                      [-1, -1, -1, 1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, 1, -1, -1, -1, -1, -1, -1, -1],
                      [1, 1, 1, -1, -1, 1, -1, 1, -1, 1],
                      [-1, -1, 1, -1, -1, 1, -1, -1, 1, -1],
                      [1, 1, 1, 1, 1, 1, -1, 1, -1, -1],
                      [1, 1, -1, 1, -1, -1, -1, 1, -1, 1],
                      [-1, 1, 1, 1, 1, 1, 1, 1, 1, -1]])
hopfield_net = HopfieldNetwork(10)
hopfield_net.train(image1)
hopfield_net.train(image2)
for noise_level in range(10):
    noisy_test_data = np.copy(test_data)
    noisy_test_data[1, :noise_level] = -1
    noisy_test_data[2, :noise_level + 1] = -1
    noisy_test_data[5, :noise_level - 1] = -1
    noisy_test_data[8, :noise_level - 1] = 1
    noisy_test_data[0, :noise_level + 1] = 1
    noisy_test_data[9, :noise_level - 1] = -1
    results = [hopfield_net.test(noisy_test_data[i], image1) for i in range(noisy_test_data.shape[0])]
    num_correct = 0
    for arr1 in image1:
        arr2: ndarray
        for i in arr1:
            for arr2 in results:
                for j in arr2:
                    if i == j:
                        num_correct += 1
    print(f"Noise level: {noise_level}, Correctly recognized: {num_correct / 10000 * 100}%")
