#!/usr/bin/env python3
# imports
from numpy import exp, transpose , diag, newaxis
from numpy.random import rand
# sigmoid
# TODO : remove static to set a specific uncenterty for each sigmoïd
class Sigmoid():
    uncertainty = 0.125
    #staticmethod
    def value(x):
        result = 1 / (1 + exp(-Sigmoid.uncertainty*x))
        return result
    #staticmethod
    def derivative(x):
        value=Sigmoid.value(x)
        result = Sigmoid.uncertainty*value/(value-1)
        return result
    pass
# perceptron
class Perceptron():
    def __init__(self,layerHeights,weightLimit=0.125):
        # initialize attributs
        self.weights=list()
        # for each layer
        for layerIndex in range(1,len(layerHeights)): #INFO : there is no weights related to input layer
            # get heights for current & previous layers
            currentHeight = layerHeights[layerIndex]
            previousHeight = layerHeights[layerIndex-1]
            # randomize layer weights
            layerWeights=(rand(currentHeight,previousHeight)-.5)*2*weightLimit
            self.weights.append(layerWeights)
            pass
        pass
    pass
    def run(self, input, training = False):
        # initialize training activation history
        if training:
            self.inputs = list()
            self.aggregations=list()
            self.outputs = list()
        # initialize current layer input
        currentInput=input
        # for each layer
        for layerIndex in range(1, len(self.weights)+1): #INFO : there is no weights related to input layer
            # get activation inputs
            layerOutput = self.activateLayer(currentInput, layerIndex, True)
            # next layer input is current layer outpout
            currentInput = layerOutput
            pass
        result = layerOutput
        return result
    # activation input : A = sum(W*I)
    def activateLayer(self, input, layerIndex, training = False):
        layerWeights = self.weights[layerIndex-1] #INFO : there is no weights related to input layer
        # compute sigmoid input
        aggregation = layerWeights.dot(input)
        # activate layer
        output = Sigmoid.value(aggregation)
        # memorize (if training)
        if training:
            self.inputs.append(input)
            self.aggregations.append(aggregation)
            self.outputs.append(output)
        # return
        return output
    # correct output layer : error = sigmoide'(aggregation) * ( expected_output - actual_output )
    def computeOutputError(self,expectedOutput):
        actualOutput = self.outputs[-1]
        aggregation = self.aggregations[-1]
        error = Sigmoid.derivative(aggregation) * (expectedOutput - actualOutput)
        self.errors[-1] = error
        pass
    # correct output layer : error = sigmoide'(aggregation) * sum(weights*previous_error)
    def computeHiddenError(self,reverseHiddenLayerIndex):
        # INFO : we start from hidden layer closest to output and move to the one closest from input
        aggregation = self.aggregations[-reverseHiddenLayerIndex-1]
        weights = self.weights[-reverseHiddenLayerIndex] #INFO : there is no weights related to input layer
        previousError = self.errors[-reverseHiddenLayerIndex] #INFO : there is no error related to input layer
        error = Sigmoid.derivative(aggregation) * weights.T.dot(previousError)
        self.errors[-reverseHiddenLayerIndex-1] = error
        pass
    pass
pass
# perceptron initialization
layerHeights=((30,24,17,10))
perceptron = Perceptron(layerHeights)
# perceptron run for training
input=tuple([round(rand()) for _ in range(layerHeights[0])])
result = perceptron.run(input, True)
print("result="+str(result))
pass
# compute output layer error
perceptron.errors=[None]*(len(perceptron.aggregations))
expectedOutput = tuple([round(rand()) for _ in range(layerHeights[-1])])
perceptron.computeOutputError(expectedOutput)
# compute hidden layer errors
hidenLayersNumber = len(perceptron.weights) - 1
for reverseHiddenLayerIndex in range(1,hidenLayersNumber+1)  : # INFO : we start from hidden layer closest to output and move to the one closest from input
    perceptron.computeHiddenError(reverseHiddenLayerIndex)
# compute new weights
for layerIndex in range(1,hidenLayersNumber+1) : # INFO : we go over hidden & output layer
    pass
pass