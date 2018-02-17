import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


def random_matrix(*dimensions, start=-1, stop=1):
    if len(dimensions) == 1:
        dimensions = dimensions + (1,)

    return (np.random.rand(*dimensions) * (stop - start)) + start
