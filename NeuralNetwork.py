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


def generateWeights(layerData, **kwargs):
    minimum, maximum = karges(kwargs, {'minimum': -1, 'maximum': 1})
    return [np.random.uniform(minimum, maximum, [layerData[1:][layer], layerData[layer] + 1]) for layer in range(len(layerData) - 1)]


def imageSplit(image, kernelShape):
        imageShape = list(image.shape)
        return np.moveaxis(np.array(np.split(np.moveaxis(np.array(np.split(image, len(image[0, 0]) / kernelShape[1], axis = 2)), 0, 1), len(image[0]) / kernelShape[0], axis = 2)), 0, 1)


class Model:
    def __init__(self):


class FullyConnected:
    def __init__(self, weights, actFunc = sigmoid):
        self.weights, self.actFunc = weights, actFunc


    def layer(self, inputs, **kwargs):
        weights, actFunc = karges(kwargs, {'weights': self.weights, 'actFunc': self.actFunc})
        return actFunc(np.tensordot(np.append(inputs, 1), weights, axis = [-1, -1]))


    def gradient(self, inputs, outputs, **kwargs):
        actFunc, actFuncDeriv = karges(kwargs, {'actFunc': sigmoid, 'actFuncDeriv': sigmoidD})
        weightedSum = self.layer(inputs, actFunc = lol, **kwargs)
        layerOutputs = actFunc(weightedSum)
        chainDerivCoef = 2 * (layerOutputs - outputs) * actFuncD(weightedSum)
        inputsGrad = np.sum(weights.transpose()[:-1] * chainDerivCoef, axis = -1)
        weightsGrad = np.outer(np.append(inputs, 1), chainDerivCoef).transpose()
        return inputsGrad, weightsGrad


