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
        return '<NeuralNetwork {}>'.format(
            ':'.join([str(s) for s in self.layer_sizes])
        )

    def feed_forward(self, input):
        """
        Passes the input through each layer, keeps track of the state.
        """
        # Transpose input if it's in the wrong shape
        if input.shape[1] != 1:
            input = input.transpose()

        for layer in self.layers:
            output = layer(input)
            yield (layer, input, output)
            input = output

    def __call__(self, input):
        """
        Just returns the last output.
        """
        *_, (_, _, output) = self.feed_forward(input)
        return output

    def train(self, input, expected_output):
        pass
