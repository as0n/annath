import numpy as np
import utils

class NeuralNetwork(object):
	def __init__(self, input_size, output_size, *args, **kwargs):
		self.n_inputs = input_size
		self.n_outputs = output_size
		self.inputs = []
		self.outputs = []
		self.synapses = []

		prev_size = input_size
		for n in args:
			self.synapses.append(utils.random_matrix(prev_size, n))
			prev_size = n
		self.synapses.append(utils.random_matrix(prev_size, output_size))

	def learn(self, inputs, outputs):
		self.inputs.append(inputs)
		self.outputs.append(outputs)

	def process(self, inputs):
		cur = inputs
		states = [cur]
		for syn in self.synapses:
			cur = utils.sigmoid(np.dot(cur, syn))
			states.append(cur)
		return states

	def train(self, n):
		X = np.array(self.inputs)
		for i in range(n):
			# Evaluation
			states = self.process(X)

			# Corrections
			last_state = states.pop()
			last_error = np.array(self.outputs) - last_state
			last_delta = last_error * utils.sigmoid(last_state, True)
			self.synapses[-1] += states[-1].T.dot(last_delta)

			prev_delta = last_delta
			for i, state in reversed(list(enumerate(states))):
				self.synapses[i] += state.T.dot(prev_delta)

				if i > 0:
					error = prev_delta.dot(self.synapses[i].T)
					prev_delta = error * utils.sigmoid(state, True)

	def error(self):
		states = self.process(np.array(self.inputs))
		error = np.array(self.outputs) - states[-1]
		return np.mean(np.abs(error))

	def apply(self, *inputs):
		return self.process(inputs)[-1]

	def apply_multiple(self, inputs):
		return [self.apply(*input) for input in inputs]
