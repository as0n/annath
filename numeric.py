from neuralnet import NeuralNetwork
import numpy as np

class Numeric(NeuralNetwork):

	def __init__(self, word=64, *args, **kwargs):
		self.word = word
		super(Numeric, self).__init__(self.word, self.word, *args, **kwargs)

	def dec2bin(self, x):
		return map(int, np.binary_repr(x, self.word))

	def bin2dec(self, y):
		return int(''.join(map(lambda x: str(int(round(x))), y)), base=2)

	def learn(self, input, output):
		super(Numeric, self).learn(self.dec2bin(input),self. dec2bin(output))

	def apply(self, input):
		tab = super(Numeric, self).apply(self.dec2bin(input))
		return self.bin2dec(tab.tolist()[0])

	def apply_multiple(self, inputs):
		return np.array([self.apply(input) for input in inputs])
