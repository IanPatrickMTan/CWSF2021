import numpy as np
import cupy as cp
import _pickle as pkl
from numba import jit


@jit(nopython=True)
def softmax(x):
	return np.exp(x - np.max(x)) / np.exp(x - np.max(x)).sum()


@jit(nopython=True)
def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def deepcopy(x):
    return pkl.loads(pkl.dumps(x))


def cost(t, v):
    return (t - v) ** 2


def sumCost(t, v):
    return np.sum((t - v) ** 2, axis = -1)


def karges(kwargs, check, default):
    if check in kwargs:
        return kwargs[check]
    else:
        return default


def generateWeights(layerData, **kwargs):
    minimum, maximum = karges(kwargs, 'minimum', -1), karges(kwargs, 'maximum', 1)
    return [np.random.uniform(minimum, maximum, [layerData[1:][layer], layerData[layer] + 1]) for layer in range(len(layerData) - 1)]


def layer(inputs, weights, **kwargs):
    return karges(kwargs, 'actFunc', sigmoid)(np.tensordot(np.append(inputs, np.full(list(inputs.shape)[:-1] + [1], 1), axis = -1), weights, axes = [[-1], [-1]]))


def neuralNetwork(inputs, weights, **kwargs):
    actFunc, finalActFunc = karges(kwargs, 'actFunc', sigmoid), karges(kwargs, 'finalActFunc', sigmoid)
    neuronOutputs = []
    layerInputs = deepcopy(inputs)
    for layerWeights in weights[:-1]:
        layerInputs = layer(layerInputs, layerWeights, actFunc = actFunc)
        neuronOutputs.append(deepcopy(layerInputs))
    
    neuronOutputs.append(layer(layerInputs, weights[-1], actFunc = finalActFunc))
    return neuronOutputs


def optimizer(inputs, weights, outputs, **kwargs):
    rawCost, learningRate, dx = karges(kwargs, 'rawCost', cost(outputs, layer(inputs, weights, **kwargs))), karges(kwargs, 'learningRate', 0.1), karges(kwargs, 'dx', 0.01)
    inputAmount = len(inputs) + 1
    dxWeights = np.eye(inputAmount).reshape([inputAmount, 1, inputAmount]) * dx + weights
    newWeights = weights - (cost(outputs, layer(inputs, dxWeights, **kwargs)) - rawCost).transpose() / dx * learningRate
    newInputs = deepcopy(inputs).astype(np.float64)
    for inputIndex, dxInput in enumerate(inputs + dx):
        newInputs[inputIndex] = dxInput
        newInputs[inputIndex] = inputs[inputIndex] - (sumCost(outputs, layer(newInputs, newWeights)) - sumCost(outputs, layer(inputs, newWeights))) / dx * learningRate
    
    return newInputs, newWeights


def train(datasetFunc, weights, **kwargs):
    costThreshold, iterLimit = karges(kwargs, 'costThreshold', 0.1), karges(kwargs, 'iterLimit', 1000)
    iterCost = 1
    newWeights = deepcopy(weights)
    for iteration in range(iterLimit):
        if iterCost <= costThreshold:
            break
        inputs, outputs = datasetFunc(iteration, **kwargs)
        newWeights = backProp(inputs, newWeights, outputs, **kwargs)
        prediction = neuralNetwork(inputs, weights, **kwargs)[-1]
        iterCost = sumCost(outputs, prediction)
        print(f'Statistics of iteration #{iteration + 1}:\n\nPrediction: {prediction}\n\nDataset Output: {outputs}\n\nCost: {iterCost}')
    return newWeights


def backProp(inputs, weights, outputs, **kwargs):
    layerInputs = neuralNetwork(inputs, weights, **kwargs)[:-1][::-1] + [inputs]
    newWeights = deepcopy(weights)[::-1]
    targetOutputs = outputs
    for layerIndex, layerWeights in enumerate(newWeights):
        print('starting layer', layerIndex)
        targetOutputs, newWeights[layerIndex] = optimizer(layerInputs[layerIndex], layerWeights, targetOutputs, **kwargs)
        print('finished layer', layerIndex)
    return newWeights[::-1]


if __name__ == '__main__':
    i = np.arange(2)
    o = np.array([1, 0, 1])
    o1 = np.array([1, 0, 0, 1, 1])
    w = generateWeights([2, 5, 3])
    w1 = deepcopy(w[0])
    nw = deepcopy(w1)