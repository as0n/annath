from utils import random_matrix, sigmoid


class Layer(object):
    def __init__(self, neurons, input_size):
        self.neurons = neurons
        self.input_size = input_size

        self.weights = random_matrix(neurons, input_size)
        self.bias = random_matrix(neurons)

    def __repr__(self):
        return '<Layer {}:{}>'.format(self.input_size, self.neurons)

    def __call__(self, input):
        return self.feed_forward(input)

    def feed_forward(self, input):
        return sigmoid(self.weights @ input + self.bias)

    def back_propagate(self, errors):
        return self.weights.transpose() @ errors
