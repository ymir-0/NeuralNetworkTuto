#!/usr/bin/env python3
# imports
from numpy import exp
from numpy.random import rand
# sigmoid
def sigmoid(x,uncertainty=0.125):
    result = 1 / (1 + exp(-uncertainty*x))
    return result
# perceptron
class Perceptron():
    def __init__(self,layerHeights,weightLimit=0.125,thresholdLimit=0.125):
        # initialize attributs
        self.weights=list()
        self.thresholds=list()
        # for each layer
        for layerIndex in range(1,len(layerHeights)): #INFO : there is no weights/thresholds related to input layer
            # get heights for current & previous layers
            currentHeight = layerHeights[layerIndex]
            previousHeight = layerHeights[layerIndex-1]
            # randomize layer weights
            layerWeights=(rand(currentHeight,previousHeight)-.5)*2*weightLimit
            self.weights.append(layerWeights)
            # randomize layer thresholds
            threshold=(rand(currentHeight)-.5)*2*thresholdLimit
            self.thresholds.append(threshold)
            pass
        pass
    pass
    def run(self, input):
        # initialize current layer input
        currentInput=input
        # for each layer
        for layerIndex in range(1, len(self.weights)+1): #INFO : there is no weights/thresholds related to input layer
            # get activation inputs
            currentActivationInputs = self.activationInputs(currentInput, layerIndex)
            currentActivationResults = self.activationResults(currentActivationInputs, layerIndex)
            currentInput = currentActivationResults
            pass
        result = currentInput
        return result
    # activation input : A = sum(W*I)
    def activationInputs(self, input, layerIndex):
        layerWeights = self.weights[layerIndex-1] #INFO : there is no weights related to input layer
        activationInputs = layerWeights.dot(input)
        return activationInputs
    # activation result : O = sigmoïd(A-threshold)
    def activationResults(self, activationInputs, layerIndex):
        layerThreshold = self.thresholds[layerIndex-1] #INFO : there is no thresholds related to input layer
        sigmoidInputs = activationInputs - layerThreshold
        activationResults = sigmoid(sigmoidInputs)
        return activationResults
    pass
pass
# perceptron initialization
layerHeights=((30,24,17,10))
perceptron = Perceptron(layerHeights)
# perceptron run
inputs=tuple([round(rand()) for _ in range(layerHeights[0])])
result = perceptron.run(inputs)
print("result="+str(result))
pass
