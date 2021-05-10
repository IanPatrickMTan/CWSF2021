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


def costDerivative(t, v):
    return np.sum(2 * (t - v), axis = -1)


def karges(kwargs, defaults):
    vars = deepcopy(defaults)
    vars.update(kwargs)
    return [vars[x] for x in list(defaults)]


def imageSplit(image, kernelShape):
    return np.moveaxis(np.array(np.split(np.moveaxis(np.array(np.split(image, len(image[0, 0]) / kernelShape[1], axis = 2)), 0, 1), len(image[0]) / kernelShape[0], axis = 2)), 0, 1)


def imageFormat(image, kernelShape):
    imageShape = list(image.shape)
    formattedImage = np.zeros([imageShape[0], imageShape[1] - kernelShape[0] + 1, imageShape[2] - kernelShape[1] + 1] + kernelShape + [imageShape[3]])
    if imageShape[1] - 2 < kernelShape[0]:
        height = kernelShape[0] - imageShape[1] - 2
    else:
        height = kernelShape[0]
    if imageShape[2] - 2 < kernelShape[1]:
        width = kernelShape[1] - imageShape[2] - 2
    else:
        width = kernelShape[1]
    for y in range(kernelShape[0]):
        for x in range(kernelShape[1]):
            formattedImage[:, y::kernelShape[0], x::kernelShape[1]] = imageSplit(image[:, y:imageShape[1] - (imageShape[1] - y) % kernelShape[0], x:imageShape[2] - (imageShape[2] - x) % kernelShape[1]], kernelShape)
    return formattedImage.reshape(imageShape[0], imageShape[1] - kernelShape[0] + 1, imageShape[2] - kernelShape[1] + 1, -1, imageShape[3])


class Model:
    def __init__(self, layers = []):
        self.layers = layers


    def evaluate(self, inputs):
        inputs = deepcopy(inputs)
        for layer in self.layers:
            inputs = deepcopy(layer.layer(inputs, **kwargs))


class convolution:
    def __init__(self, kernels, kernelShape, activation = sigmoid, activationDerivative = sigmoidD, cost = cost, costDerivative = costDerivative):
        self.kernels, self.kernelShape, self.activation, self.activationDerivative, self.cost, self.costDerivative = deepcopy(kernels), kernelShape, activation, activationDerivative, cost, costDerivative


    def layer(self, inputs, **kwargs):
        kernels, kernelShape, activation = karges(kwargs, {'kernels': self.kernels, 'kernelShape': self.kernelShape, 'activation': self.activation})
        inputs = imageFormat(inputs, kernelShape)
        biases = np.full(list(inputs.shape[:-2]) + [1] + [inputs.shape[-1]], 1)
        inputs = np.append(inputs, biases, axis = -2)
        weightedSum = activation(np.einsum('abcde, fgh -> fabce', inputs, kernels))
        return weightedSum.reshape([-1] + list(weightedSum.shape)[2:])


class dense:
    def __init__(self, weights, activation = sigmoid, activationDerivative = sigmoidD, cost = cost, costDerivative = costDerivative):
        self.weights, self.activation, self.activationDerivative, self.cost, self.costDerivative = deepcopy(weights), activation, activationDerivative, cost, costDerivative


    def layer(self, inputs, **kwargs):
        weights, activation = karges(kwargs, {'weights': self.weights, 'activation': self.activation})
        return activation(np.tensordot(np.append(inputs, 1), weights, axes = [-1, -1]))


    def gradient(self, inputs, outputs, **kwargs):
        weights, activation, activationDerivative, cost, costDerivative = karges(kwargs, {'weights': self.weights, 'activation': self.activation, 'activationDerivative': self.activationDerivative, 'cost': self.cost, 'costDerivative': self.costDerivative})
        weightedSum = self.layer(inputs, activation = lol, **kwargs)
        layerOutputs = activation(weightedSum)
        chainDerivCoef = costDerivative(outputs, layerOutputs) * activationDerivative(weightedSum)
        inputsGrad = np.sum(weights.transpose()[:-1] * chainDerivCoef, axis = -1)
        weightsGrad = np.outer(np.append(inputs, 1), chainDerivCoef).transpose()
        return inputsGrad, weightsGrad


