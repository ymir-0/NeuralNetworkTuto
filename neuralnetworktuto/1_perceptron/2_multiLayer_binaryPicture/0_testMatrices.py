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
            self.aggregations=list()
            self.outputs = list()
        # initialize current layer input
        currentInput=input
        # for each layer
        for layerIndex in range(1, len(self.weights)+1): #INFO : there is no weights related to input layer
            # get activation inputs
            layerOutput = self.activateLayer(currentInput, layerIndex, True)
            # compute & memorize sigmoid input (if training)
            if training:
                self.outputs.append(layerOutput)
            # next layer input is current layer outpout
            currentInput = layerOutput
            pass
        result = layerOutput
        return result
    # activation input : A = sum(W*I)
    def activateLayer(self, input, layerIndex, training = False):
        layerWeights = self.weights[layerIndex-1] #INFO : there is no weights related to input layer
        # compute & memorize sigmoid input (if training)
        aggregation = layerWeights.dot(input)
        if training:
            self.aggregations.append(aggregation)
        # activate layer
        output = Sigmoid.value(aggregation)
        # return
        return output
    # correct output layer : error = sigmoide'(E) * ( T - S )
    def outputError(self,expectedOutput): # INFO : T=expectedOutput ; pas=correctionStep
        # compute weights correction step
        actualOutput = self.outputs[-1] # S
        outputAggregation = self.aggregations[-1] # E
        outputError = Sigmoid.derivative(outputAggregation) * (expectedOutput - actualOutput)
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
# correct output layer
expectedOutput = tuple([round(rand()) for _ in range(layerHeights[-1])])
outputError = perceptron.outputError(expectedOutput)
