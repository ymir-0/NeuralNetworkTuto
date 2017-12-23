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
    def __init__(self, trainings):
        # randomize initial neuron network
        originalTrainingIndexes = tuple(trainings.keys())
        self.neuronsNumber = len(originalTrainingIndexes)  # we want one neuron for each input test data
        self.neuronsInputSize = len(trainings[originalTrainingIndexes[0]])  # we can length of first training input
        self.randomizeNetwork()
        # initialize correction step
        self.currentCorrectionStep = Perceptron.correctionStep
        # assume network is not trained
        trained = False
        # training as many time as needed
        actualLoopNumber = 0
        while not trained:
            #
            trained = self.trainRandomizedFullSet(originalTrainingIndexes)
            # check loops number
            actualLoopNumber = actualLoopNumber + 1
            if actualLoopNumber >= Perceptron.computeLimitLoop:
                raise Exception("Sorry, random choices are too long to adjust. Better to retry")
            # adjust correction step
                self.currentCorrectionStep = self.currentCorrectionStep * Perceptron.correctionFactor
    def randomizeNetwork(self):
        self.neurons = list()
        for neuronIndex in range(0, self.neuronsNumber):
            self.neurons.append(Neuron(self.neuronsInputSize))
    def trainRandomizedFullSet(self,originalTrainingIndexes):
        # for each random input data
        currentTrainingValues = list(originalTrainingIndexes)
        shuffle(currentTrainingValues)
        currentTrainingValues = tuple(currentTrainingValues)
        for currentTrainingValue in currentTrainingValues:
            trained = self.computeCurrentTrainingValue(currentTrainingValue)
        # return
        return trained
    def computeCurrentTrainingValue(self,currentTrainingValue):
        # assume network is trained
        trained = True
        # construct expected output
        expectedOutput = [0] * self.neuronsNumber
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
                # input data must be active
                if currentInputValue == 1:
                    # correct each neuron
                    self.correctAllNeurons( impliedNeuron, currentInputIndex,expectedOutput, actualOutput)
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
    def correctAllNeurons(self,impliedNeuron,currentInputIndex,expectedOutput,actualOutput):
        # for each neuron
        for currentNeuronIndex in range(0, self.neuronsNumber):
                # compute related correction value
                neuronDifference = expectedOutput[currentNeuronIndex] - actualOutput[currentNeuronIndex]
                # correct this neuron if needed
                if neuronDifference != 0:
                    self.correctSingleNeuron(impliedNeuron, currentInputIndex, neuronDifference)
    def correctSingleNeuron(self,impliedNeuron,currentInputIndex,neuronDifference):
            impliedNeuron.thresholdedWeights[currentInputIndex] = impliedNeuron.thresholdedWeights[currentInputIndex] + self.currentCorrectionStep * neuronDifference
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
