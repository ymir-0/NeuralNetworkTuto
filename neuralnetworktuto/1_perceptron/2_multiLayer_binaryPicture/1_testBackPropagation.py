#!/usr/bin/env python3
''' examples from :
 - https://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks
 - https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
 - http://www.anyflo.com/bret/cours/rn/rn5.htm#exemple
'''
# imports
from numpy import exp, newaxis, zeros, array, sum
from numpy.ma import size
from numpy.random import rand
from os import linesep, sep, listdir, makedirs
from os.path import realpath, join, exists
from random import shuffle
from shutil import rmtree
from collections import Iterable
from enum import Enum, unique
from copy import deepcopy
# sigmoid
class Sigmoid():
    @staticmethod
    def value(variables, uncertainties=1, dilatations=1, offsets=0):
        value = dilatations / (1 + exp(-variables * uncertainties)) + offsets
        return value
    @staticmethod
    # INFO : we compute the derivative from : value = sigmoïd(variables)
    def derivativeFromValue(value, uncertainties=1, dilatations=1):
        derivative = dilatations * uncertainties * value * (1 - value)
        return derivative
# training draft
class TrainingDraft():
    def __init__(self,input,weightsBiasInput,output):
        self.input = input
        self.weightsBiasInput = weightsBiasInput
        self.output = output
# perceptron
# INFO : can not defined a common parameters enumeration : https://docs.python.org/3/library/enum.html#restricted-subclassing-of-enumerations
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
    def passForward(self,input,training=False):
        # compute ouput
        weightsBiasInput = self.weights.dot(input) + self.biases
        output = Sigmoid.value(weightsBiasInput, self.uncertainties, self.dilatations, self.offsets)
        # initialize training draft (if justified by context)
        if training:
            self.trainingDraft = TrainingDraft(input, weightsBiasInput, output)
        return output
    def passBackward(self,expectedOutput=None,differentialErrorWeightsBiasInput=None,previousLayerWeights=None):
        # get differential error on layer regarding output or hidden one
        if expectedOutput is not None:
            differentialErrorLayer = self.differentialErrorOutput(expectedOutput)
        else:
            differentialErrorLayer = self.differentialErrorHidden(differentialErrorWeightsBiasInput,previousLayerWeights)
        # compute new weights
        differentialOutputWeightsBiasInput = Sigmoid.derivativeFromValue(array([self.trainingDraft.output]), self.uncertainties, array([self.dilatations]))
        # INFO : new differential error on layer will be used on next computation
        newDifferentialErrorWeightsBiases = (differentialErrorLayer * differentialOutputWeightsBiasInput).T
        differentialErrorWeights = newDifferentialErrorWeightsBiases * self.trainingDraft.input
        # TODO : set learning rate 0.5 has variable (and add inertia)
        # INFO : old weights will be used on next computation
        oldWeights = self.weights
        self.weights = oldWeights - 0.5 * differentialErrorWeights
        # compute new biases
        newBiases = self.biases - 0.5 * newDifferentialErrorWeightsBiases.T
        self.biases = tuple(newBiases[0])
        # compute new dilatations
        differentialOutputDilatations = Sigmoid.value(self.trainingDraft.weightsBiasInput, self.uncertainties)
        differentialErrorDilatations = differentialErrorLayer * differentialOutputDilatations
        newDilatations = self.dilatations - 0.5 * differentialErrorDilatations
        self.dilatations = newDilatations
        # compute new uncertainties
        differentialOutputUncertainties = Sigmoid.derivativeFromValue(array([self.uncertainties]), self.trainingDraft.output, array([self.dilatations]))
        differentialErrorUncertainties = differentialErrorLayer * differentialOutputUncertainties
        newUncertainties = self.uncertainties - 0.5 * differentialErrorUncertainties
        self.uncertainties = tuple(newUncertainties[0])
        # compute new offsets
        newOffsets = self.offsets - 0.5 * differentialErrorLayer
        self.offsets = newOffsets
        # discard training draft
        del self.trainingDraft
        # return
        return newDifferentialErrorWeightsBiases, oldWeights
    # get differential error on output layer
    def differentialErrorOutput(self,expectedOutput):
        differentialError = self.trainingDraft.output - expectedOutput
        return differentialError
    # get differential error on hidden layer
    def differentialErrorHidden(self,differentialErrorWeightsBiasInput,previousLayerWeights):
        differentialErrors = differentialErrorWeightsBiasInput * previousLayerWeights
        differentialError = sum(differentialErrors, 0)
        return differentialError
    pass
class Perceptron():
    # TODO : add methods to manipulate perceptron : remove weight between 2 neurons, remove specific neuron, edit specific neuron meta parameter
    def __init__(self,**parameters):
        # layerHeights,weights=None,weightLimit=0.125,biasLimit=1,uncertainties=1,dilatations=1,offsets=0
        # TODO : check weights dimensions consistancy if planned (not random) : at least 2 layers (input/output) and matrices dimensions
        # TODO : check extra parameters dimensions consistancy if enumerates
        # initialize layers
        self.layers = list()
        # test planned weights
        plannedWeights = WeightParameters.WEIGHTS.value in parameters
        # set layer number
        # INFO : we do not create the input layer because it will be the input vector to forward pass with
        layerNumber = len(parameters[PerceptronParameters.WEIGHTS.value]) if plannedWeights else parameters[PerceptronParameters.LAYER_HEIGHTS.value] - 1
        # initialize meta parameters
        metaParameters = MetaParameters.defaultValues()
        # set meta parameters has enumerates
        for name in MetaParameters.enumerate():
            value = parameters[name]  if name in parameters else metaParameters[name]
            if not isinstance(value, Iterable):
                value = [value] * layerNumber
            metaParameters[name] = tuple(value)
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
            self.layers.append(layer)
        # TODO : tuple layers after training ?
    def passForward(self,input,training=False):
        # INFO : next input is actual output
        inputOutput = input
        for layer in self.layers:
            inputOutput = layer.passForward(inputOutput,training)
        return inputOutput
    def passForwardBackward(self,input,expectedOutput):
        # pass forward
        actualOutput = self.passForward(input=input,training=True)
        # compute total error
        outputError = ( ( expectedOutput - actualOutput ) ** 2 ) / 2
        totalError = sum(outputError,0)
        # pass backward
        self.passBackward(expectedOutput)
        # return
        return totalError
    def passBackward(self,expectedOutput):
        # pass on output
        layer = self.layers[-1]
        differentialErrorWeightsBiasInput, previousLayerWeights = layer.passBackward(expectedOutput=expectedOutput)
        # pass on hidden layers
        for hiddenLayerIndex in range(2, len(self.layers)+1):
            layer = self.layers[-hiddenLayerIndex]
            differentialErrorWeightsBiasInput, previousLayerWeights = layer.passBackward(differentialErrorWeightsBiasInput=differentialErrorWeightsBiasInput,previousLayerWeights=previousLayerWeights)
    pass
# ***** 1 hidden layer , 2 neurons on each layer
# perceptron initialization
weights=((
    array(((
        ((0.15,0.2)),
        ((0.25,0.3)),
    ))),
    array(((
        ((0.4,0.45)),
        ((0.5,0.55)),
    ))),
))
biases=((0.35,0.6))
perceptronModel = Perceptron(weights=weights,biases=biases,uncertainties=.99)
# make many forward & backward pass
perceptron = deepcopy(perceptronModel)
input = ((0.05,0.1))
expectedOutput = ((0.01,0.99))
for loopNumber in range(6):
    totalError = perceptron.passForwardBackward(input, expectedOutput)
output = perceptron.passForward(input)
print("total error = " + str(totalError))
print("pass forward output = " + str(output))
'''
# ***** 1 hidden layer , 3 neurons on input&output layer, 2 neurons on hidden layer
# perceptron initialization
weights=((
    array(((
        ((0.5, 0.3, 0.1)),
        ((0.3, 0.2, 0.1)),
    ))),
    array(((
        ((0.1,0.2)),
        ((0.3,0.4)),
        ((0.5, 0.6)),
    ))),
))
perceptron = Perceptron(weights=weights)
# forward pass
input = ((1,2,3))
output =  perceptron.passForward(input)
print("expected pass forward output =\n[0.5564 0.6302 0.6984]")
print("actual pass forward output =\n" + str(output))
# complete backward pass
perceptron.passBackward(((0.1,0.3,0.7)))
#print("expected pass backward output =\n[[0.35891648 0.408666186]\n[0.51130127 0.561370121]]")
print("actual pass backward output =\n" + str(perceptron.layers[-1].weights))
#print("expected pass backward hidden =\n[[0.4946 0.2892 0.0837]\n[0.2896 0.1791 0.0687]]")
print("actual pass backward hidden =\n" + str(perceptron.layers[-2].weights))
'''
pass
