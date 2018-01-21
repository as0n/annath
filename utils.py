import numpy as np


# Sigmoid function : R -> [0-1]
def sigmoid(x, deriv=False):
    return x*(1-x) if deriv else 1/(1+np.exp(-x))


def random_matrix(*dimensions, start=-1, stop=1):
    return (np.random.rand(*dimensions) * (stop - start)) + start
