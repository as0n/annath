from utils import sigmoid_deriv
import numpy as np
import random


class Teacher(object):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
        self.examples = []

    def add_example(self, input, output):
        self.examples.append(
            (input, output)
        )

    def get_random_example(self):
        return random.choice(self.examples)

    def train(self, network):
        (input, expected_output) = self.get_random_example()

        steps = list(network.feed_forward(input))

        steps.reverse()
        error = expected_output - steps[0][2]

        for (layer, input, output) in steps:
            gradient = self.learning_rate * np.multiply(
                sigmoid_deriv(output),
                error
            )
            delta = gradient @ input.transpose()

            # Magic
            layer.weights += delta
            layer.bias += gradient

            error = layer.back_propagate(error)
