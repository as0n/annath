from layer import Layer


class Network(object):
    def __init__(self, *layer_sizes):
        self.layer_sizes = layer_sizes
        self.layers = []

        input_size = layer_sizes[0]
        for layer_size in layer_sizes[1:]:
            self.layers.append(Layer(layer_size, input_size))
            input_size = layer_size

    def __repr__(self):
        return '<NeuralNetwork {}:{}>'.format(
            self.layer_sizes[0],
            self.layer_sizes[-1]
        )

    def __call__(self, input):
        for layer in self.layers:
            input = layer(input)

        return input

    def train(self, input, expected_output):
        pass
