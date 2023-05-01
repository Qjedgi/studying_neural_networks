from collections import defaultdict

import numpy as np


def normalize_data(data_for_normalize):
    norm_data = data_for_normalize.copy()
    min_vals = np.min(norm_data, axis=0)
    max_vals = np.max(norm_data, axis=0)
    norm_data = (norm_data - min_vals) / (max_vals - min_vals)
    return norm_data


class KohonenNetwork:

    def __init__(self, input_size, output_size, init_learning_rate):
        self.epoch = None
        self.input_size = input_size
        self.output_size = output_size
        self.init_learning_rate = init_learning_rate
        self.weights = np.random.rand(output_size, input_size)

    def get_best_matching_unit(self, sample):
        distances = np.linalg.norm(self.weights - sample, axis=1)
        return np.argmin(distances)

    def update_weights(self, bmu, sample, learning_rate):
        for i in range(self.output_size):
            distance = np.linalg.norm(bmu - i)
            neighborhood_radius = self.output_size * np.exp(-self.epoch / 6)
            if distance <= neighborhood_radius:
                self.weights[i] += learning_rate * (sample - self.weights[i])

    def train(self, data, num_epochs):
        norm_data = normalize_data(data)
        learning_rates = [self.init_learning_rate - i * 0.05 for i in range(num_epochs)]
        for epoch in range(num_epochs):
            self.epoch = epoch
            for sample in norm_data:
                bmu = self.get_best_matching_unit(sample)
                learning_rate = learning_rates[epoch]
                self.update_weights(bmu, sample, learning_rate)

    def predict(self, data):
        norm_data = normalize_data(data)
        predictions = np.zeros(len(data), dtype=np.int64)
        for i, sample in enumerate(norm_data):
            bmu = self.get_best_matching_unit(sample)
            predictions[i] = bmu
        return predictions


lastnames = ["Vardanyan", "Gorbunov", "Gumenyuk", "Egorov", "Zakharova", "Ivanova", "Ishonina",
             "Klimchuk", "Lisovsky", "Netreba", "Ostapova", "Pashkova", "Popov", "Sazon",
             "Stepanenko", "Terentieva", "Titov", "Chernova", "Chetkin", "Shevchenko"]
data = np.array([
    [1, 1, 1, 60, 79, 60, 72, 63, 1.00],
    [2, 1, 0, 60, 61, 30, 5, 17, 0.00],
    [3, 0, 0, 60, 61, 30, 66, 58, 0.00],
    [4, 1, 1, 85, 78, 72, 70, 85, 1.25],
    [5, 0, 1, 65, 78, 60, 67, 65, 1.00],
    [6, 0, 1, 60, 78, 77, 81, 60, 1.25],
    [7, 0, 1, 55, 79, 56, 69, 72, 0.00],
    [8, 1, 0, 55, 56, 50, 56, 60, 0.00],
    [9, 1, 0, 55, 60, 21, 64, 50, 0.00],
    [10, 1, 0, 60, 56, 30, 16, 17, 0.00],
    [11, 0, 1, 85, 89, 85, 92, 85, 1.75],
    [12, 0, 1, 60, 88, 76, 66, 60, 1.25],
    [13, 1, 0, 55, 64, 0, 9, 50, 0.00],
    [14, 0, 1, 80, 83, 62, 72, 72, 1.25],
    [15, 1, 0, 55, 10, 3, 8, 50, 0.00],
    [16, 0, 1, 60, 67, 57, 64, 50, 0.00],
    [17, 1, 1, 75, 98, 86, 82, 85, 1.50],
    [18, 0, 1, 85, 85, 81, 85, 72, 1.25],
    [19, 1, 1, 80, 56, 50, 69, 50, 0.00],
    [20, 1, 0, 55, 60, 30, 8, 60, 0.00]
], dtype=np.float64)
res = defaultdict(list)
clustering_data = data[:, 1:8]
input_size = clustering_data.shape[1]
output_size = 4
init_learning_rate = 0.30
num_epochs = 6
kohonen_network = KohonenNetwork(input_size, output_size, init_learning_rate)
kohonen_network.train(clustering_data, num_epochs)
print("Predicted clusters:")
predictions = kohonen_network.predict(clustering_data)
for key in predictions:
    for val in lastnames:
        res[key].append(val)
        lastnames.remove(val)
for key, val in dict(res).items():
    print(key, ': ', val)

