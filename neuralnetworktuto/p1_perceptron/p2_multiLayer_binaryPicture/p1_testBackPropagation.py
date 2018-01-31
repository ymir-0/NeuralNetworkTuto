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
from statistics import mean
from matplotlib.pyplot import plot, xticks, yticks, title , xlabel , ylabel, grid, figure, legend, tick_params, savefig, show
# sigmoid
# TODO : create an abstract class for all future functions
# TODO : compute with spark each method
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
    @staticmethod
    def defaultValues():
        defaultValues = {
            "weightLimit" : 1,
        }
        return defaultValues
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
            # TODO : compute with spark 'weights'
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
            metaParameters[name] = value
        # set meta parameters has enumerates
        for name , value in metaParameters.items():
            setattr(self, name , value)
    def passForward(self,input,training=False):
        # compute ouput
        # TODO : compute with spark 'weightsBiasInput'
        weightsBiasInput = self.weights.dot(input) + self.biases
        output = Sigmoid.value(weightsBiasInput, self.uncertainties, self.dilatations, self.offsets)
        # initialize training draft (if justified by context)
        if training:
            self.trainingDraft = TrainingDraft(input, weightsBiasInput, output)
        return tuple(output)
    def passBackward(self,expectedOutput=None,differentialErrorWeightsBiasInput=None,previousLayerWeights=None, learningRate=0.5, metaParametersUpdate=((MetaParameters.BIASES.value))):
        # TODO : compute with spark each parameter
        # TODO : add inertia
        # get differential error on layer regarding output or hidden one
        if expectedOutput is not None:
            differentialErrorLayer = self.differentialErrorOutput(expectedOutput)
        else:
            differentialErrorLayer = self.differentialErrorHidden(differentialErrorWeightsBiasInput,previousLayerWeights)
        # compute new weights
        newDifferentialErrorWeightsBiases, oldWeights = self.computeNewWeights(differentialErrorLayer, learningRate)
        # compute new extra parameters
        # TODO : set extra parameters to update
        if MetaParameters.BIASES.value in metaParametersUpdate:
            self.computeNewBiases(newDifferentialErrorWeightsBiases, learningRate)
        if MetaParameters.UNCERTAINTIES.value in metaParametersUpdate:
            self.computeNewUncertainties(differentialErrorLayer, learningRate)
        if MetaParameters.DILATATIONS.value in metaParametersUpdate:
            self.computeNewDilatations(differentialErrorLayer, learningRate)
        if MetaParameters.OFFSETS.value in metaParametersUpdate:
            self.computeNewOffsets(differentialErrorLayer, learningRate)
        # discard training draft
        del self.trainingDraft
        # return
        return newDifferentialErrorWeightsBiases, oldWeights
    # get differential error on output layer
    def differentialErrorOutput(self,expectedOutput):
        # TODO : compute with spark 'differentialError'
        differentialError = self.trainingDraft.output - expectedOutput
        return differentialError
    # get differential error on hidden layer
    def differentialErrorHidden(self,differentialErrorWeightsBiasInput,previousLayerWeights):
        # TODO : compute with spark 'differentialError'
        differentialErrors = differentialErrorWeightsBiasInput * previousLayerWeights
        differentialError = sum(differentialErrors, 0)
        return differentialError
    def computeNewWeights(self,differentialErrorLayer, learningRate=0.5):
        differentialOutputWeightsBiasInput = Sigmoid.derivativeFromValue(array([self.trainingDraft.output]), self.uncertainties, array([self.dilatations]))
        # INFO : new differential error on layer will be used on next computation
        newDifferentialErrorWeightsBiases = (differentialErrorLayer * differentialOutputWeightsBiasInput).T
        differentialErrorWeights = newDifferentialErrorWeightsBiases * self.trainingDraft.input
        # INFO : old weights will be used on next computation
        oldWeights = self.weights
        self.weights = oldWeights - learningRate * differentialErrorWeights
        return newDifferentialErrorWeightsBiases, oldWeights
    def computeNewBiases(self,differentialErrorWeightsBiases, learningRate=0.5):
        newBiases = self.biases - learningRate * differentialErrorWeightsBiases.T
        self.biases = newBiases[0]
    def computeNewUncertainties(self,differentialErrorLayer, learningRate=0.5):
        differentialOutputUncertainties = Sigmoid.derivativeFromValue(array([self.uncertainties]), self.trainingDraft.output, array([self.dilatations]))
        differentialErrorUncertainties = differentialErrorLayer * differentialOutputUncertainties
        newUncertainties = self.uncertainties - learningRate * differentialErrorUncertainties
        self.uncertainties = newUncertainties[0]
    def computeNewDilatations(self,differentialErrorLayer, learningRate=0.5):
        differentialOutputDilatations = Sigmoid.value(self.trainingDraft.weightsBiasInput, self.uncertainties)
        differentialErrorDilatations = differentialErrorLayer * differentialOutputDilatations
        newDilatations = self.dilatations - learningRate * differentialErrorDilatations
        self.dilatations = newDilatations
    def computeNewOffsets(self,differentialErrorLayer, learningRate=0.5):
        newOffsets = self.offsets - learningRate * differentialErrorLayer
        self.offsets = newOffsets
    def freeze(self):
        # freeze weights
        frozenWeights = list()
        for weightsColumn in self.weights:
            frozenWeights.append(tuple(weightsColumn))
        frozenWeights = tuple(frozenWeights)
        self.weights = array(frozenWeights)
        # freeze meta parameters
        self.biases = tuple(self.biases)
        self.uncertainties = tuple(self.uncertainties)
        self.dilatations = tuple(self.dilatations)
        self.offsets = tuple(self.offsets)
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
        # set layer number & weights limit
        # INFO : we do not create the input layer because it will be the input vector to forward pass with
        layerNumber = len(parameters[PerceptronParameters.WEIGHTS.value]) if plannedWeights else len(parameters[PerceptronParameters.LAYER_HEIGHTS.value]) -1
        if PerceptronParameters.WEIGHT_LIMIT.value not in parameters:
            parameters[PerceptronParameters.WEIGHT_LIMIT.value] = PerceptronParameters.defaultValues()[PerceptronParameters.WEIGHT_LIMIT.value]
        if not isinstance(parameters[PerceptronParameters.WEIGHT_LIMIT.value], Iterable):
            parameters[PerceptronParameters.WEIGHT_LIMIT.value] = [parameters[PerceptronParameters.WEIGHT_LIMIT.value]]*layerNumber
        parameters[PerceptronParameters.WEIGHT_LIMIT.value] = tuple(parameters[PerceptronParameters.WEIGHT_LIMIT.value])
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
    def passForward(self,input,training=False):
        # TODO : add a pipe to cascade passes (only if not training)
        # INFO : next input is actual output
        inputOutput = input
        for layer in self.layers:
            inputOutput = layer.passForward(inputOutput,training)
        return inputOutput
    def train(self,sequences,maximumLoopNumber=int(1e5),minimumMeanError=1e-10,errorsRecordNumber=100, metaParametersUpdate=((MetaParameters.BIASES.value))):
        # initialize errors
        errors=dict()
        errors["loopNumbers"]=list()
        # initialize error conter
        # TODO : simplify error mesures : send all mesure as bulk in a DB, and unstack table later (regarding needed number of results)
        keepErrorMeasures = errorsRecordNumber is not None and errorsRecordNumber > 0
        if keepErrorMeasures:
            for expectedOutput in sequences.values():
                errors[expectedOutput]=list()
            errorsMeasureStep = maximumLoopNumber / errorsRecordNumber
            errorsMeasureLoop = int(errorsMeasureStep)
        # train has necessary
        for loopNumber in range(maximumLoopNumber):
            currentErrors = self.trainRandomized(sequences, metaParametersUpdate)
            # keep error measure if necessary
            if keepErrorMeasures and loopNumber == errorsMeasureLoop:
                errors["loopNumbers"].append(loopNumber)
                [errors[expectedOutput].append(error) for expectedOutput, error in currentErrors.items()]
                # set next error measurement step
                errorsMeasureLoop = int(errorsMeasureLoop+errorsMeasureStep)
            # test if error is sufficient
            meanError = mean(currentErrors.values())
            if meanError <= minimumMeanError :
                break
        # freeze when trained
        if keepErrorMeasures:
            for expectedOutput, error in errors.items():
                errors[expectedOutput]=tuple(error)
        self.freeze()
        # return
        return errors
    def trainRandomized(self,sequences, metaParametersUpdate=((MetaParameters.BIASES.value))):
        # initialize errors
        errors = dict()
        # randomize sequence
        randomizedInputs = list(sequences.keys())
        shuffle(randomizedInputs)
        randomizedInputs = tuple(randomizedInputs)
        # run forward & backward for each training input / expected output
        for input in randomizedInputs:
            expectedOutput = sequences[input]
            error = self.passForwardBackward(input, expectedOutput, metaParametersUpdate)
            errors[expectedOutput] = error
        # return
        return errors
    def passForwardBackward(self,input,expectedOutput, metaParametersUpdate=((MetaParameters.BIASES.value))):
        # pass forward
        actualOutput = self.passForward(input=input,training=True)
        # compute total error
        outputError = ( ( expectedOutput - array([actualOutput]) ) ** 2 ) / 2
        totalError = sum(outputError)
        # pass backward
        self.passBackward(expectedOutput, metaParametersUpdate)
        # return
        return totalError
    def passBackward(self,expectedOutput, metaParametersUpdate=((MetaParameters.BIASES.value))):
        # pass on output
        layer = self.layers[-1]
        differentialErrorWeightsBiasInput, previousLayerWeights = layer.passBackward(expectedOutput=expectedOutput, metaParametersUpdate=metaParametersUpdate)
        # pass on hidden layers
        for hiddenLayerIndex in range(2, len(self.layers)+1):
            layer = self.layers[-hiddenLayerIndex]
            differentialErrorWeightsBiasInput, previousLayerWeights = layer.passBackward(differentialErrorWeightsBiasInput=differentialErrorWeightsBiasInput,previousLayerWeights=previousLayerWeights, metaParametersUpdate=metaParametersUpdate)
    def freeze(self):
        # freeze layer details
        for layer in self.layers:
            layer.freeze()
        # freeze all layers
        self.layers = tuple(self.layers)
    pass
pass