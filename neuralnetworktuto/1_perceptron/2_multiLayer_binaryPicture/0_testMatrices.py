#!/usr/bin/env python3
# imports
from numpy import exp, newaxis
from numpy.random import rand
from os import linesep, sep, listdir, makedirs
from os.path import realpath, join, exists
from random import shuffle
# contants
CURRENT_DIRECTORY = realpath(__file__).rsplit(sep, 1)[0]
# tools classes
def readTraining():
    inputDirectory = join(CURRENT_DIRECTORY,"input")
    # initialize data
    trainings = list()
    # for each data file
    for dataFileShortName in listdir(inputDirectory):
        # contruct expected output
        expectedOutput = [0]*10
        digit = int(dataFileShortName.split(".")[0])
        expectedOutput[digit] = 1
        # read it
        dataFileFullName = join(inputDirectory, dataFileShortName)
        dataFile = open(dataFileFullName)
        rawData = dataFile.read()
        dataFile.close()
        # construct image
        dataPivot = rawData.replace(linesep, "")
        image = [int(pixel) for pixel in dataPivot]
        # fill data
        training=((tuple(image),tuple(expectedOutput)))
        trainings.append(training)
    # return
    return tuple(trainings)
# sigmoid
class Sigmoid():
    # TODO : remove static to set a specific uncertainty for each sigmoïd
    uncertainty = 1e-2
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
        # INFO : there is no weights related to input layer
        for layerIndex in range(1,len(layerHeights)):
            self.randomizeLayer(layerIndex,weightLimit)
            pass
        pass
    def randomizeLayer(self, layerIndex,weightLimit):
        # get heights for current & previous layers
        currentHeight = layerHeights[layerIndex]
        previousHeight = layerHeights[layerIndex-1]
        # randomize layer weights
        layerWeights=(rand(currentHeight,previousHeight)-.5)*2*weightLimit
        self.weights.append(layerWeights)
        pass
    pass
    def runAllLayers(self, input):
        # initialize training activation history
        if hasattr(self,"training"):
            self.inputs = list()
            self.aggregations=list()
            self.outputs = list()
        # initialize current layer input
        currentInput=input
        # for each hidden & output layer
        # INFO : there is no weights related to input layer
        for layerIndex in range(0, len(self.weights)):
            # next layer input is current layer outpout
            currentInput = self.runSpecificLayer(currentInput, layerIndex)
            pass
        pass
    def runSpecificLayer(self, input, layerIndex):
        # get activation inputs
        # INFO : there is no weights related to input layer
        layerWeights = self.weights[layerIndex]
        # compute sigmoid
        aggregation = layerWeights.dot(input)
        output = Sigmoid.value(aggregation)
        # memorize (if training)
        if hasattr(self,"training"):
            self.inputs.append(input)
            self.aggregations.append(aggregation)
            self.outputs.append(output)
        # next layer input is current layer outpout
        return output
        pass
    def executeCompleteTrainingStep (self,data):
        # enable training state
        self.training = None
        # try each data
        randomizedData=list(data)
        shuffle(randomizedData)
        randomizedData=tuple(randomizedData)
        for singleData in randomizedData:
            self.executeOneTrainingStep(singleData)
        # remove training state
        del self.training
        del self.inputs
        del self.aggregations
        del self.outputs
        self.weights=tuple(self.weights) # TODO : tuplize each sub-array
        pass
    def executeOneTrainingStep (self,data):
        # parse training data
        input = data[0]
        expectedOutput =  data[1]
        # run one training over all layers
        perceptron.runAllLayers(input)
        # compute output layer error
        perceptron.errors = [None] * (len(perceptron.aggregations))
        perceptron.computeOutputError(expectedOutput)
        # compute hidden layer errors
        perceptron.computeAllHiddenErrors()
        # compute new weights
        perceptron.computeAllNewWeights()
        pass
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
        # INFO : we start from hidden layer closest to output and move to the one closest from input
        for reverseHiddenLayerIndex in range(1,len(self.weights)):
            self.computeSpecificHiddenError(reverseHiddenLayerIndex)
        pass
    def computeSpecificHiddenError(self,reverseHiddenLayerIndex):
        # INFO : we start from hidden layer closest to output and move to the one closest from input
        aggregation = self.aggregations[-reverseHiddenLayerIndex-1]
        # INFO : there is no weights nor error related to input layer
        layerWeights = self.weights[-reverseHiddenLayerIndex]
        previousError = self.errors[-reverseHiddenLayerIndex]
        error = Sigmoid.derivative(aggregation) * layerWeights.T.dot(previousError)
        self.errors[-reverseHiddenLayerIndex-1] = error
        pass
    def computeAllNewWeights(self):
        # we only work on hidden & output layers
        # INFO : we start from hidden layer closest to input and move to the output one
        for layerIndex in range(0,len(self.weights)):
            self.computeSpecificNewLayerWeights(layerIndex)
        pass
    def computeSpecificNewLayerWeights(self,layerIndex):
        # INFO : there is no weights related to input layer
        currentLayerWeights = self.weights[layerIndex]
        lambda_ = 1 # TODO : make it a parameter
        error = self.errors[layerIndex]
        input = self.inputs[layerIndex]
        newLayerWeights = currentLayerWeights + lambda_ * error[newaxis].T * input
        self.weights[layerIndex] = newLayerWeights # TODO : add inertia https://fr.wikipedia.org/wiki/R%C3%A9tropropagation_du_gradient
    pass
pass
# perceptron initialization
layerHeights=((30,24,17,10))
perceptron = Perceptron(layerHeights)
# train perceptron
trainings = readTraining()
perceptron.executeCompleteTrainingStep (trainings)
# TODO : train for a complete set
# TODO : train until ready
# TODO : compute statistics
# TODO : optimize uncertainty
pass