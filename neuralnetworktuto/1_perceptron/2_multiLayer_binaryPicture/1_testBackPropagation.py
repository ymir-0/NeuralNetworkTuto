#!/usr/bin/env python3
''' examples from :
 - https://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks
 - https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
 - https://becominghuman.ai/back-propagation-is-very-simple-who-made-it-complicated-97b794c97e5c
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
    def passForward(self,input):
        weightsBiasInput = self.weights.dot(input) + self.biases
        output = self.dilatations / (1 + exp( -weightsBiasInput * self.uncertainties)) + self.offsets
        return output
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
    def passForward(self,input):
        # TODO : use self.historyInputsOutputs only if learning
        self.historyInputsOutputs=list()
        # INFO : next input is actual output
        inputOutput = input
        for layer in self.layers:
            self.historyInputsOutputs.append(inputOutput)
            inputOutput = layer.passForward(inputOutput)
        self.historyInputsOutputs.append(inputOutput)
        return inputOutput
    def passBackward(self,expectedOutput):
        # initialize new weights
        newLayersWeights = list()
        # pass on output
        newWeights = self.passBackwardOutput(expectedOutput)
        newLayersWeights.append(newWeights)
        # pass on hidden layers
        for hiddenLayerIndex in range(0, len(self.layers)-1):
            newWeights = self.passBackwardHidden()
            newLayersWeights.append(newWeights)
        # set new weights on all layers
        pass
    def passBackwardOutput(self,expectedOutput):
        # cast to array to array to avoid issues
        actualOutput = self.historyInputsOutputs[-1]
        differentialErrorOutput = actualOutput - expectedOutput
        lastLayer = self.layers[-1]
        differentialOutputWeightsBiasInput = array([lastLayer.dilatations]) * lastLayer.uncertainties * actualOutput * (1-array([actualOutput]))
        self.differentialErrorWeightsBiasInput = (differentialErrorOutput * differentialOutputWeightsBiasInput).T
        differentialErrorWeights= self.differentialErrorWeightsBiasInput * self.historyInputsOutputs[-2]
        # TODO : set learning rate 0.5 has variable (and add inertia)
        newWeights = lastLayer.weights - 0.5 * differentialErrorWeights
        return newWeights
    def passBackwardHidden(self):
        differentialErrorsOutput = self.differentialErrorWeightsBiasInput * self.layers[-1].weights
        differentialErrorOutput = sum(differentialErrorsOutput,0)
        hiddenLayer = self.layers[-2]
        hiddenInput = self.historyInputsOutputs[-2]
        differentialOutputWeightsBiasInput =  array([hiddenLayer.dilatations]) * hiddenLayer.uncertainties * hiddenInput * (1-array([hiddenInput]))
        differentialErrorWeights= (differentialErrorOutput * differentialOutputWeightsBiasInput).T * self.historyInputsOutputs[-3]
        # TODO : set learning rate 0.5 has variable (and add inertia)
        newWeights = hiddenLayer.weights - 0.5 * differentialErrorWeights
        return newWeights
    pass
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
perceptron = Perceptron(weights=weights,biases=biases)
# forward pass
input = ((0.05,0.1))
output =  perceptron.passForward(input)
print("expected pass forward output =\n[0.75136507 0.772928465]")
print("actual pass forward output =\n" + str(output))
# backward pass on output
newWeights = perceptron.passBackwardOutput(((0.01,0.99)))
print("expected pass backward output =\n[[0.35891648 0.408666186]\n[0.51130127 0.561370121]]")
print("actual pass backward output =\n" + str(newWeights))
# backward pass on hiddenLayer
newWeights = perceptron.passBackwardHidden()
print("expected pass backward hidden =\n[[0.149780719 0.19956143]\n[0.24975114 0.29950229]]")
print("actual pass backward hidden =\n" + str(newWeights))
# complete backward pass
newWeights = perceptron.passBackward(((0.01,0.99)))
print("expected pass backward output =\n[[0.35891648 0.408666186]\n[0.51130127 0.561370121]]")
print("actual pass backward output =\n" + str(newWeights))
pass
