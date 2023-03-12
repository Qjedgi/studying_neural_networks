import random
class Layer:
    def __init__(self, prev_neuron_count, cur_neuron_count, input_signals, step_learn):
        self.out = []
        self.step_learn = step_learn
        self.step_y = step_learn / 3
        self.inp = input_signals
        self.prev_neuron_count = prev_neuron_count
        self.cur_neuron_count = cur_neuron_count
        self.weights = []
        for _ in range(cur_neuron_count):
            weight = []
            for i in range(prev_neuron_count):
                weight.append(random.random())
            self.weights.append(weight)
    def weight_change(self):
        for i in range(self.cur_neuron_count):
            for j in range(self.prev_neuron_count):
                self.weights[i][j] = self.step_learn * self.inp[j][0] * self.inp[j][1] * self.out[i] + self.weights[i][
                    j] * (1 - self.step_y)
    def calculate(self):
        self.out = []
        for i in range(self.cur_neuron_count):
            summary = 0
            for j in range(self.prev_neuron_count):
                summary += self.inp[j][0] * self.weights[i][j] + self.inp[j][1] * self.weights[i][j]
            self.out.append(self.activate_function(summary))
    def activate_function(self, value):
        if value >= 0:
            return 1
        else:
            return 0
    def __str__(self):
        s = ""
        for ww in self.weights:
            s += f'{ww}\n'
        return s
if __name__ == "__main__":
    x = [[0.97, 0.2], [1, 0],
         [-0.72, 0.7], [-0.67, 0.74],
         [-0.8, 0.6], [0, -1],
         [0.2, -0.97], [-0.3, -0.95]]
    layer = Layer(4, 4, x, 0.5)
    print(layer, "\n")
    layer.calculate()
    layer.weight_change()
    print(layer)
