import numpy as np
import _pickle as pkl
from numba import jit
from datetime import datetime


def lol(x):
    return x


def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def sigmoidD(x):
	return sigmoid(x) * (1 - sigmoid(x))


def deepcopy(x):
    return pkl.loads(pkl.dumps(x))


def cost(t, v):
    return np.sum((t - v) ** 2, axis = -1)


def karges(kwargs, defaults):
    vars = deepcopy(defaults)
    vars.update(kwargs)
    return [vars[x] for x in list(defaults)]


def imageSplit(image, kernelShape):
        imageShape = list(image.shape)
        return np.moveaxis(np.array(np.split(np.moveaxis(np.array(np.split(image, len(image[0, 0]) / kernelShape[1], axis = 2)), 0, 1), len(image[0]) / kernelShape[0], axis = 2)), 0, 1)


class FullyConnected:
    def __init__(self, weights, actFunc):
        self.weights, self.actFunc = weights, actFunc
