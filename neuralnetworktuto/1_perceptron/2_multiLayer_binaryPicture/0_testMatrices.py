#!/usr/bin/env python3
# imports
from numpy import exp, transpose , diag, newaxis, array
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
    def runAllLayers(self, input, training = False):
        # initialize training activation history
        if training:
            self.inputs = list()
            self.aggregations=list()
            self.outputs = list()
        # initialize current layer input
        currentInput=input
        # for each hidden & output layer
        for layerIndex in range(0, len(self.weights)): #INFO : there is no weights related to input layer
            # next layer input is current layer outpout
            currentInput = self.runSpecificLayer(currentInput, layerIndex, training)
            pass
        pass
    def runSpecificLayer(self, input, layerIndex, training):
        # get activation inputs
        layerWeights = self.weights[layerIndex] #INFO : there is no weights related to input layer
        # compute sigmoid input
        aggregation = layerWeights.dot(input)
        # activate layer
        output = Sigmoid.value(aggregation)
        # memorize (if training)
        if training:
            self.inputs.append(input)
            self.aggregations.append(aggregation)
            self.outputs.append(output)
        # next layer input is current layer outpout
        return output
        pass
    # activation input : A = sum(W*I)
    # correct output layer : error = sigmoide'(aggregation) * ( expected_output - actual_output )
    def computeOutputError(self,expectedOutput):
        # we only work on output layer
        actualOutput = self.outputs[-1]
        aggregation = self.aggregations[-1]
        error = Sigmoid.derivative(aggregation) * (expectedOutput - actualOutput)
        self.errors[-1] = error
        pass
    # correct output layer : error = sigmoide'(aggregation) * sum(weights*previous_error)
    def computeAllHiddenErrors(self):
        # we only work on hidden layers
        hidenOutputLayersNumber = len(self.weights)
        for reverseHiddenLayerIndex in range(1,hidenOutputLayersNumber):  # INFO : we start from hidden layer closest to output and move to the one closest from input
            self.computeSpecificHiddenError(reverseHiddenLayerIndex)
        pass
    def computeSpecificHiddenError(self,reverseHiddenLayerIndex):
        # INFO : we start from hidden layer closest to output and move to the one closest from input
        aggregation = self.aggregations[-reverseHiddenLayerIndex-1]
        layerWeights = self.weights[-reverseHiddenLayerIndex] #INFO : there is no weights related to input layer
        previousError = self.errors[-reverseHiddenLayerIndex] #INFO : there is no error related to input layer
        error = Sigmoid.derivative(aggregation) * layerWeights.T.dot(previousError)
        self.errors[-reverseHiddenLayerIndex-1] = error
        pass
    def computeAllNewWeights(self):
        # we only work on hidden & output layers
        hidenOutputLayersNumber = len(self.weights)
        for layerIndex in range(0,hidenOutputLayersNumber):  # INFO : we start from hidden layer closest to input and move to the output one
            self.computeSpecificNewLayerWeights(layerIndex)
        pass
    def computeSpecificNewLayerWeights(self,layerIndex):
        currentLayerWeights = self.weights[layerIndex]  # INFO : there is no weights related to input layer
        lambda_ = 1 # TODO : make it a parameter
        error = self.errors[layerIndex]
        input = self.inputs[layerIndex]
        newLayerWeights = currentLayerWeights + lambda_ * error[newaxis].T * input
        self.weights[layerIndex] = newLayerWeights
    pass
pass
# perceptron initialization
layerHeights=((30,24,17,10))
perceptron = Perceptron(layerHeights)
# perceptron run for training
input=tuple([round(rand()) for _ in range(layerHeights[0])])
perceptron.runAllLayers(input, True)
# compute output layer error
perceptron.errors=[None]*(len(perceptron.aggregations))
expectedOutput = tuple([round(rand()) for _ in range(layerHeights[-1])])
perceptron.computeOutputError(expectedOutput)
# compute hidden layer errors
perceptron.computeAllHiddenErrors()
# compute new weights
perceptron.computeAllNewWeights()
pass