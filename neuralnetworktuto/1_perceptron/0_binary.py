#!/usr/bin/env python3
# imports
from numpy import heaviside, array, append
from numpy.random import rand
from random import shuffle
from copy import copy
# neuron
class Neuron():
    def __init__(self,neuronsInputSize):
        weights=2*rand(neuronsInputSize)/32767-1 # TODO: explain magic number
        threshold=rand(1) # we assume threshold to be an weight so it can be adjust and condidere 'global threshold'=0
        self.thresholdedWeights=append(weights,-threshold)
        pass
    def activate(self,inputs):
        # sum weighted input
        thresholdedInputs = append(inputs, 1)
        weightedInputs = self.thresholdedWeights.dot(array(thresholdedInputs).transpose())
        # compute & return OUT
        output = heaviside(weightedInputs, 1)
        return output
# neuron network
class Perceptron():
    computeLimitLoop=10000 # sometimes, random choices are too long to adjust. better to retry
    correctionStep=0.1 # TODO: explain magic number
    correctionFactor=1 # TODO: explain magic number
    def __init__(self,trainings):
        # randomize initial neuron network
        originalTrainingIndexes=tuple(trainings.keys())
        neuronsNumber=len(originalTrainingIndexes) # we want one neuron for each input test data
        neuronsInputSize=len(trainings[originalTrainingIndexes[0]]) # we can length of first training input
        self.randomizeNetwork(neuronsNumber,neuronsInputSize)
        # initialize correction step
        currentCorrectionStep=Perceptron.correctionStep
        # assume network is not trained
        trained=False
        # training as many time as needed
        actualLoopNumber=0
        while not trained:
            #
            trained = self.trainRandomizedFullSet(originalTrainingIndexes,neuronsNumber,currentCorrectionStep)
            # check loops number
            actualLoopNumber = actualLoopNumber + 1
            if actualLoopNumber >= Perceptron.computeLimitLoop:
                raise Exception("Sorry, random choices are too long to adjust. Better to retry")
            # adjust correction step
            currentCorrectionStep = currentCorrectionStep * Perceptron.correctionFactor
    def randomizeNetwork(self,neuronsNumber,neuronsInputSize):
        self.neurons=list()
        for neuronIndex in range(0,neuronsNumber):
            self.neurons.append(Neuron(neuronsInputSize))
    def trainRandomizedFullSet(self,originalTrainingIndexes,neuronsNumber,currentCorrectionStep):
        # for each random input data
        currentTrainingValues = list(originalTrainingIndexes)
        shuffle(currentTrainingValues)
        currentTrainingValues = tuple(currentTrainingValues)
        for currentTrainingValue in currentTrainingValues:
            trained = self.computeCurrentTrainingValue(neuronsNumber, currentTrainingValue, currentCorrectionStep)
        # return
        return trained
    def computeCurrentTrainingValue(self,neuronsNumber,currentTrainingValue,currentCorrectionStep):
        # assume network is trained
        trained = True
        # construct expected output
        expectedOutput = [0] * neuronsNumber
        expectedOutput[currentTrainingValue] = 1
        expectedOutput = tuple(expectedOutput)
        # get implied neuron
        impliedNeuron = self.neurons[currentTrainingValue]
        # get corresponding training
        correspondingTraining = trainings[currentTrainingValue]
        # compute actual output for actual training data
        actualOutput = self.execute(correspondingTraining)
        # correct weights if necessary
        if actualOutput != expectedOutput:
            # network is not trained
            trained = False
            # for each input data
            thresholdedInputs = enumerate(append(correspondingTraining, 1))
            for currentInputIndex, currentInputValue in thresholdedInputs:
                # correct each neuron
                self.correctAllNeurons(neuronsNumber, impliedNeuron, currentInputIndex, currentInputValue,expectedOutput, actualOutput, currentCorrectionStep)
        # return
        return trained
    def execute(self,inputs):
        # initialise outputs
        outputs=list()
        # compute each output (for each neuron)
        for neuronIndex in range(0, len(self.neurons)):
            currentOutput = self.neurons[neuronIndex].activate(inputs)
            outputs.append(currentOutput)
        # return
        return tuple(outputs)
    def correctAllNeurons(self,neuronsNumber,impliedNeuron,currentInputIndex,currentInputValue,expectedOutput,actualOutput,currentCorrectionStep):
        # for each neuron
        for currentNeuronIndex in range(0, neuronsNumber):
            # input data must be active
            if currentInputValue == 1:
                # correct this neuron
                self.correctSingleNeuron(impliedNeuron,currentInputIndex,expectedOutput,actualOutput,currentCorrectionStep,currentNeuronIndex)
    def correctSingleNeuron(self,impliedNeuron,currentInputIndex,expectedOutput,actualOutput,currentCorrectionStep,currentNeuronIndex):
        # compute related correction value
        neuronDifference = expectedOutput[currentNeuronIndex] - actualOutput[currentNeuronIndex]
        # correct if needed
        if neuronDifference != 0:
            impliedNeuron.thresholdedWeights[currentInputIndex] = impliedNeuron.thresholdedWeights[currentInputIndex] + currentCorrectionStep * neuronDifference
# set training data
completeTrainings={
    0:
        tuple([1, 1, 1, 1, 1,
               1, 0, 0, 0, 1,
               1, 0, 0, 0, 1,
               1, 0, 0, 0, 1,
               1, 0, 0, 0, 1,
               1, 1, 1, 1, 1]),
    1:
        tuple([0, 0, 1, 0, 0,
               0, 0, 1, 0, 0,
               0, 0, 1, 0, 0,
               0, 0, 1, 0, 0,
               0, 0, 1, 0, 0,
               0, 0, 1, 0, 0]),
    2:
        tuple([1, 1, 1, 1, 1,
               0, 0, 0, 0, 1,
               1, 1, 1, 1, 1,
               1, 0, 0, 0, 0,
               1, 0, 0, 0, 0,
               1, 1, 1, 1, 1]),
    3:
        tuple([1, 1, 1, 1, 1,
               0, 0, 0, 0, 1,
               1, 1, 1, 1, 1,
               0, 0, 0, 0, 1,
               0, 0, 0, 0, 1,
               1, 1, 1, 1, 1] ),
    4:
        tuple([0, 1, 0, 0, 0,
               0, 1, 0, 0, 0,
               0, 1, 0, 0, 0,
               0, 1, 0, 1, 0,
               0, 1, 1, 1, 1,
               0, 0, 0, 1, 0] ),
    5:
        tuple([1, 1, 1, 1, 1,
               1, 0, 0, 0, 0,
               1, 0, 0, 0, 0,
               1, 1, 1, 1, 1,
               0, 0, 0, 0, 1,
               1, 1, 1, 1, 1] ),
    6:
        tuple([1, 1, 1, 1, 1,
               1, 0, 0, 0, 0,
               1, 0, 0, 0, 0,
               1, 1, 1, 1, 1,
               1, 0, 0, 0, 1,
               1, 1, 1, 1, 1] ),
    7:
        tuple([1, 1, 1, 1, 0,
               0, 0, 0, 1, 0,
               0, 0, 0, 1, 0,
               0, 0, 1, 1, 1,
               0, 0, 0, 1, 0,
               0, 0, 0, 1, 0] ),
    8:
        tuple([1, 1, 1, 1, 1,
               1, 0, 0, 0, 1,
               1, 1, 1, 1, 1,
               1, 0, 0, 0, 1,
               1, 0, 0, 0, 1,
               1, 1, 1, 1, 1]),
    9:
        tuple([1, 1, 1, 1, 1,
               1, 0, 0, 0, 1,
               1, 1, 1, 1, 1,
               0, 0, 0, 0, 1,
               0, 0, 0, 0, 1,
               1, 1, 1, 1, 1] ),
}
trainings={
    0: completeTrainings[0],
    1: completeTrainings[1],
    2: completeTrainings[2],
    3: completeTrainings[3],
    4: completeTrainings[4],
    5: completeTrainings[5],
    #6: completeTrainings[6],
    #7: completeTrainings[7],
    #8: completeTrainings[8],
    #9: completeTrainings[9],
}
# train neuron network
perceptron=Perceptron(trainings)
# test results
print("N ->  0    1    2    3    4    5")
for inputValue, inputImage in trainings.items():
    outputValues=perceptron.execute(inputImage)
    print(str(inputValue) + " -> " + str(outputValues))
