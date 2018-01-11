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
    def run(self, input, training = False):
        # initialize training activation history
        if training:
            self.sigmoidInputs=list()
        # initialize current layer input
        currentInput=input
        # for each layer
        for layerIndex in range(1, len(self.weights)+1): #INFO : there is no weights/thresholds related to input layer
            # get activation inputs
            currentActivationInput = self.activationInputs(currentInput, layerIndex)
            currentActivationOutput = self.activationResults(currentActivationInput, layerIndex, training)
            # next layer input is current layer outpout
            currentInput = currentActivationOutput
            pass
        result = currentActivationOutput
        return result
    # activation input : A = sum(W*I)
    def activationInputs(self, input, layerIndex):
        layerWeights = self.weights[layerIndex-1] #INFO : there is no weights related to input layer
        activationInputs = layerWeights.dot(input)
        return activationInputs
    # activation result : O = sigmoïd(A-threshold)
    def activationResults(self, activationInput, layerIndex, training = False):
        # get thresholds
        layerThreshold = self.thresholds[layerIndex-1] #INFO : there is no thresholds related to input layer
        # compute & memorize sigmoid input (if training)
        sigmoidInput= activationInput - layerThreshold
        if training:
            self.sigmoidInputs.append(sigmoidInput)
        # activate layer
        activationResults = Sigmoid.value(sigmoidInput)
        # return
        return activationResults
    # correct output layer : delta = 2 * pas * sigmoide'(E) * ( T - S )
    def correctOutputLayer(self,expectedOutput,actualOutput,correctionStep=1): # INFO : S=actualOutput ; T=expectedOutput ; pas=correctionStep
        # compute weights correction step
        outputSigmoidInput = self.sigmoidInputs[-1] # E
        delta = 2 * correctionStep * Sigmoid.derivative(outputSigmoidInput) * (expectedOutput - actualOutput)
        test0 = self.sigmoidInputs[-2]
        test1 = delta[newaxis].T * test0
        test3 = self.weights[-1] + test1
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
perceptron.correctOutputLayer(expectedOutput,result)
