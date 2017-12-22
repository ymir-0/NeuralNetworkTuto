#!/usr/bin/env python3
# imports
from numpy import heaviside, array, append
from numpy.random import rand
from random import shuffle
from copy import copy
# neuron
class Neuron():
    def __init__(self,neuronsInputSize):
        weights=rand(neuronsInputSize)
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
    pass
# neuron network
class Perceptron():
    computeLimitLoop=10000 # sometimes, random choices are too long to adjust. better to retry
    correctionStep=0.9
    correctionFactor=0.5
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
            # assume network is trained
            trained=True
            # for each random input data
            currentTrainingValues=list(originalTrainingIndexes)
            shuffle(currentTrainingValues)
            currentTrainingValues=tuple(currentTrainingValues)
            for currentTrainingIndex,currentTrainingValue in enumerate(currentTrainingValues):
                # get expected output
                expectedOutput=[0]*neuronsNumber
                expectedOutput[currentTrainingIndex]=1
                expectedOutput=tuple(expectedOutput)
                # get implied neuron
                impliedNeuron=self.neurons[currentTrainingIndex]
                # get corresponding training
                correspondingTraining=trainings[currentTrainingValue]
                # compute actual output for actual training data
                actualOutput=self.execute(correspondingTraining)
                # correct weights if necessary
                if actualOutput!=expectedOutput:
                    # network is not trained
                    trained = False
                    # for input data
                    thresholdedInputs = append(correspondingTraining, 1)
                    for currentInputIndex, currentInputValue in enumerate(thresholdedInputs):
                        # for each neuron
                        for currentNeuronIndex in range(0,neuronsNumber):
                            # input data must be active
                            if currentInputValue==1:
                                # compute related correction value
                                neuronDifference=expectedOutput[currentNeuronIndex]-actualOutput[currentNeuronIndex]
                                # correct if needed
                                if neuronDifference!=0:
                                    impliedNeuron.thresholdedWeights[currentInputIndex] = impliedNeuron.thresholdedWeights[currentInputIndex] + currentCorrectionStep*neuronDifference
                                pass
                            pass
                        pass
                    pass
                pass
            # check loops number
            actualLoopNumber = actualLoopNumber + 1
            if actualLoopNumber >= Perceptron.computeLimitLoop:
                raise Exception("Sorry, random choices are too long to adjust. Better to retry")
            # adjust correction step
            currentCorrectionStep=currentCorrectionStep*Perceptron.correctionFactor
            pass
        #
        pass
    def randomizeNetwork(self,neuronsNumber,neuronsInputSize):
        self.neurons=list()
        for neuronIndex in range(0,neuronsNumber):
            self.neurons.append(Neuron(neuronsInputSize))
            pass
        pass
    def execute(self,inputs):
        # initialise outputs
        outputs=list()
        # compute each output (for each neuron)
        for neuronIndex in range(0, len(self.neurons)):
            currentOutput = self.neurons[neuronIndex].activate(inputs)
            outputs.append(currentOutput)
        # return
        return tuple(outputs)
    pass
# set training data
completeTrainings={
    0:
        tuple([1, 1, 1, 1,
               1, 0, 0, 1,
               1, 0, 0, 1,
               1, 0, 0, 1,
               1, 0, 0, 1,
               1, 1, 1, 1] ),
    1:
        tuple([0, 0, 1, 1,
               0, 1, 1, 1,
               1, 0, 1, 0,
               0, 0, 1, 0,
               0, 0, 1, 0,
               0, 1, 1, 1]),
    2:
        tuple([1, 1, 1, 1,
               0, 0, 0, 1,
               0, 0, 1, 0,
               0, 1, 0, 0,
               1, 0, 0, 0,
               1, 1, 1, 1]),
    3:
        tuple([1, 1, 1, 1,
               0, 0, 0, 1,
               0, 0, 0, 1,
               1, 1, 1, 1,
               0, 0, 0, 1,
               1, 1, 1, 1] ),
    4:
        tuple([0, 0, 1, 0,
               0, 1, 1, 0,
               1, 0, 1, 0,
               1, 1, 1, 1,
               0, 0, 1, 0,
               0, 0, 1, 0] ),
    5:
        tuple([1, 1, 1, 1,
               1, 0, 0, 0,
               1, 1, 1, 1,
               0, 0, 0, 1,
               0, 0, 0, 1,
               1, 1, 1, 1] ),
    6:
        tuple([1, 1, 1, 1,
               1, 0, 0, 0,
               1, 1, 1, 1,
               1, 0, 0, 1,
               1, 0, 0, 1,
               1, 1, 1, 1] ),
    7:
        tuple([1, 1, 1, 1,
               0, 0, 0, 1,
               0, 0, 1, 0,
               1, 1, 1, 1,
               0, 1, 0, 0,
               1, 0, 0, 0] ),
    8:
        tuple([1, 1, 1, 1,
               1, 0, 0, 1,
               1, 0, 0, 1,
               1, 1, 1, 1,
               1, 0, 0, 1,
               1, 1, 1, 1]),
    9:
        tuple([1, 1, 1, 1,
               1, 0, 0, 1,
               1, 0, 0, 1,
               1, 1, 1, 1,
               0, 0, 0, 1,
               1, 1, 1, 1] ),
}
testDigits=[0,1]
trainings={
    testDigits[0]: completeTrainings[testDigits[0]],
    testDigits[1]: completeTrainings[testDigits[1]],
}
# train neuron network
perceptron=Perceptron(trainings)
# test results
print("N ->  0    1")
for inputValue, inputImage in trainings.items():
    outputValues=perceptron.execute(inputImage)
    print(str(inputValue) + " -> " + str(outputValues))
    pass
pass