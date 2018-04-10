#!/usr/bin/env python3
''' examples from :
 - https://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks
 - https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
 - http://www.anyflo.com/bret/cours/rn/rn5.htm#exemple
'''
# imports
from numpy import exp, array, sum, mean
from numpy.ma import size
from numpy.random import rand
from random import shuffle
from collections import Iterable
from enum import Enum, unique
from csv import writer
# sigmoid
# TODO : create an abstract class for all future functions
# TODO : compute with spark each method
# TODO : add pre/post run fct° to normalize or scale I/O
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
    def passBackward(self,expectedOutput=None,differentialErrorWeightsBiasInput=None,previousLayerWeights=None, learningRate=0.5):
        # TODO : compute with spark each parameter
        # TODO : add inertia
        # get differential error on layer regarding output or hidden one
        if expectedOutput is not None:
            differentialErrorLayer = self.differentialErrorOutput(expectedOutput)
        else:
            differentialErrorLayer = self.differentialErrorHidden(differentialErrorWeightsBiasInput,previousLayerWeights)
        # compute new weights & biases
        newDifferentialErrorWeightsBiases, oldWeights = self.computeNewWeights(differentialErrorLayer, learningRate)
        self.computeNewBiases(newDifferentialErrorWeightsBiases, learningRate)
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
        # TODO : optionaly correct oter metaparameters (offset, dilatation, ...)
        # INFO : old weights will be used on next computation
        oldWeights = self.weights
        self.weights = oldWeights - learningRate * differentialErrorWeights
        return newDifferentialErrorWeightsBiases, oldWeights
    def computeNewBiases(self,differentialErrorWeightsBiases, learningRate=0.5):
        newBiases = self.biases - learningRate * differentialErrorWeightsBiases.T
        self.biases = newBiases[0]
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
    # INFO : maximumTime in seconds. 86400 secods = 1 day
    def train(self,sequences,trainingReportName):
        # initialize training context
        trained=False
        loopCounter=0
        imagesNumber=len(sequences)
        # write report
        with open(trainingReportName, 'a') as trainingReportFile:
            reportWriter = writer(trainingReportFile, delimiter=';')
            reportWriter.writerow((("IMAGES_NUMBER", "LOOP_COUNTER", "MEAN_ERROR","IMAGES_ERROR_NUMBER")))
            # train as many as necessary
            while not trained:
                # train once
                currentErrors = self.trainRandomized(sequences)
                imagesErrorNumber=self.checkTraining(sequences)
                trained=imagesErrorNumber==0
                # fill report
                meanError = mean(currentErrors)
                reportWriter.writerow(((imagesNumber,loopCounter,meanError,imagesErrorNumber)))
                # next loop
                loopCounter+=1
                pass
            pass
        pass
    pass
    def trainRandomized(self,sequences):
        # initialize errors
        errors = list()
        # randomize sequence
        randomizedSequence = list(sequences)
        shuffle(randomizedSequence)
        randomizedSequence = tuple(randomizedSequence)
        # run forward & backward for each training input / expected output
        for index,data in enumerate(randomizedSequence):
            input=data["image"]
            expectedOutput = data["label"]
            error = self.passForwardBackward(input, expectedOutput)
            errors.append(error)
        # return
        return errors
    def passForwardBackward(self,input,expectedOutput):
        # pass forward
        actualOutput = self.passForward(input=input,training=True)
        # compute total error
        outputError = ( ( expectedOutput - array([actualOutput]) ) ** 2 ) / 2
        totalError = sum(outputError)
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
    def checkTraining(self,sequences):
        # initialize errors
        errors = 0
        # run forward & backward for each training input / expected output
        for index,data in enumerate(sequences):
            input=data["image"]
            expectedOutput = data["label"]
            expectedDigit = expectedOutput.index(1)
            actualOutput = self.passForward(input)
            actualDigit = actualOutput.index(max(actualOutput))
            error = 0 if expectedDigit==actualDigit else 1
            errors+=error
        # return
        return errors
        pass
    pass
pass
