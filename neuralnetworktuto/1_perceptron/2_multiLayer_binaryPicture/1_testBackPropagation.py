#!/usr/bin/env python3
''' examples from :
 - https://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks
 - https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
'''
# imports
from numpy import exp, newaxis, zeros
from numpy.ma import size
from numpy.random import rand
from os import linesep, sep, listdir, makedirs
from os.path import realpath, join, exists
from random import shuffle
from shutil import rmtree
from collections import Iterable
from enum import Enum, unique
# contants
CURRENT_DIRECTORY = realpath(__file__).rsplit(sep, 1)[0]
OUTPUT_DIRECTORY = join(CURRENT_DIRECTORY,"output")
# tools classes
class Logger():
    completeLog=""
    @staticmethod
    def append(level, message):
        Logger.completeLog=Logger.completeLog+" "*(4*level)+message+linesep
    @staticmethod
    def flush():
        logFile = open(join(OUTPUT_DIRECTORY,"training.log"),"wt")
        logFile.write(Logger.completeLog)
        logFile.close()
# sigmoid
def sigmoid(input,uncertainty=1,dilatation=1,offset=0):
    # INFO : bias is already included in previous aggregation step
    # TODO : set uncertainty/dilatation/offset for a layer
    output = dilatation / (1 + exp(-uncertainty*input)) + offset
    return output
    pass
# perceptron
@unique
class PerceptronParameters(Enum):
    WEIGHTS="weights"
    LAYER_HEIGHTS="layerHeights"
    WEIGHT_LIMIT="weightLimit"
@unique
class WeightParameters(Enum):
    WEIGHTS="weights"
    CURRENT_HEIGHT="currentHeight"
    PREVIOUS_HEIGHT="previousHeight"
    WEIGHT_LIMIT="weightLimit"
@unique
class MetaParameters(Enum):
    BIASES="biases"
    UNCERTAINTIES="uncertainties"
    DILATATIONS="dilatations"
    OFFSETS="offsets"
    @staticmethod
    def enumerate():
        parameters = [parameter.value for parameter in MetaParameters]
        return tuple(parameters)
    @staticmethod
    def defaultValues():
        defaultValues = {
            "biases" : 0,
            "uncertainties" : 1,
            "dilatations" : 1,
            "offsets" : 0,
        }
        return defaultValues
class Layer():
    def __init__(self,**parameters):
        # TODO : check extra parameters dimensions consistancy if enumerates (must match layer height)
        # initialize weights
        weights=parameters.get(WeightParameters.WEIGHTS.value, None)
        # randomize weights if necessary
        if weights is None:
            currentHeight = parameters.get(WeightParameters.CURRENT_HEIGHT.value)
            previousHeight = parameters.get(WeightParameters.PREVIOUS_HEIGHT.value)
            weightLimit = parameters.get(WeightParameters.WEIGHT_LIMIT.value)
            weights = (rand(currentHeight, previousHeight) - .5) * 2 * weightLimit
        self.weights = weights
        height = size(self.weights,0)
        # initialize meta parameters
        metaParameters = MetaParameters.defaultValues()
        # initialize meta parameters
        for name in MetaParameters.enumerate():
            value = parameters[name] if name in parameters else metaParameters[name]
            if not isinstance(value, Iterable):
                value = [value] * height
            metaParameters[name] = tuple(value)
        # set meta parameters has enumerates
        for name , value in metaParameters.items():
            setattr(self, name , value)
        pass
    def passForward(self,input):
        sigmoidInput = self.weights.dot(input) + self.biases
        output = sigmoid(sigmoidInput)
        return output
    pass
class Perceptron():
    # TODO : add methods to manipulate perceptron : remove weight between 2 neurons, remove specific neuron, edit specific neuron meta parameter
    def __init__(self,**parameters):
        # layerHeights,weights=None,weightLimit=0.125,biasLimit=1,uncertainties=1,dilatations=1,offsets=0
        # TODO : check weights dimensions consistancy if planned (not random) : at least 2 layers (input/output) and matrices dimensions
        # TODO : check extra parameters dimensions consistancy if enumerates
        # initialize layers
        layers = list()
        # test planned weights
        plannedWeights = WeightParameters.WEIGHTS.value in parameters
        # set layer number
        # INFO : we do not create the input layer because it will be the input vector to forward pass with
        layerNumber = len(parameters[WeightParameters.WEIGHTS.value]) if plannedWeights else layerHeights - 1
        # initialize meta parameters
        metaParameters = MetaParameters.defaultValues()
        # set meta parameters has enumerates
        for name in MetaParameters.enumerate():
            value = parameters[name]  if name in parameters else metaParameters[name]
            if not isinstance(value, Iterable):
                value = [value] * layerNumber
            metaParameters[name] = tuple(value)
            pass
        # for each layer
        for layerIndex in range(layerNumber):
            # initialize layer parameters
            layerParameters=dict()
            # fill weights parameters (regarding planned xor random)
            if plannedWeights:
                layerParameters[WeightParameters.WEIGHTS.value] = parameters[PerceptronParameters.WEIGHTS.value][layerIndex]
            else :
                # INFO : we take in account input layer height
                layerParameters[WeightParameters.CURRENT_HEIGHT.value] = parameters[PerceptronParameters.LAYER_HEIGHTS.value][layerIndex+1]
                layerParameters[WeightParameters.PREVIOUS_HEIGHT.value] = parameters[PerceptronParameters.LAYER_HEIGHTS.value][layerIndex]
                layerParameters[WeightParameters.WEIGHT_LIMIT.value] = parameters[PerceptronParameters.WEIGHT_LIMIT.value][layerIndex]
            # fill meta parameters
            for name, value in metaParameters.items():
                layerParameters[name] = metaParameters[name][layerIndex]
            # create layer
            layer = Layer(**layerParameters)
            layers.append(layer)
            pass
        # tuple layers
        self.layers = tuple(layers)
        # set
        pass
    def passForward(self,input):
        # INFO : next input is actual output
        inputOutput = input
        for layer in self.layers:
            inputOutput = layer.passForward(inputOutput)
        return inputOutput
    pass
pass
# empty output folder
if exists(OUTPUT_DIRECTORY):
    rmtree(OUTPUT_DIRECTORY)
makedirs(OUTPUT_DIRECTORY)
# perceptron initialization
layerHeights=((2,2,2))
weights=list()
weights.append(zeros((2,2)))
weights.append(zeros((2,2)))
weights[0][0][0]=0.15
weights[0][0][1]=0.2
weights[0][1][0]=0.25
weights[0][1][1]=0.3
weights[1][0][0]=0.4
weights[1][0][1]=0.45
weights[1][1][0]=0.5
weights[1][1][1]=0.55
biases=((0.35,0.6))
perceptron = Perceptron(layerHeights=layerHeights,weights=weights,biases=biases)
# compute forward pass
input = ((0.05,0.1))
output =  perceptron.passForward(input)
# flush logs
Logger.flush()
pass