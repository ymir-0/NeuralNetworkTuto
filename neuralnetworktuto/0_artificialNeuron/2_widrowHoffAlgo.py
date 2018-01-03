#!/usr/bin/env python3
# imports
from numpy import heaviside, array, append
from numpy.random import rand
from os import linesep, sep
from os.path import join, realpath
# contants
CURRENT_DIRECTORY = realpath(__file__).rsplit(sep, 1)[0]
# tools class
class Logger():
    completeLog=""
    @staticmethod
    def append(level, message):
        Logger.completeLog=Logger.completeLog+" "*(4*level)+message+linesep
    @staticmethod
    def flush():
        logFile = open(join(CURRENT_DIRECTORY,"training.log"),"wt")
        logFile.write(Logger.completeLog)
        logFile.close()
# neurons
class Neuron():
    computeLimitLoop=20 # sometimes, random choices are too long to adjust. better to retry
    def __init__(self,trainings):
        # initiate neuron
        adjustmentStep=.01
        weightsNumber=len(tuple(trainings.keys())[0]) # we use the length of 1st training data to get neurons number
        # INFO : found with a dichotomy between 1 and 0 for doing some loops
        threshold=rand()*.6+.1
        weights=rand(weightsNumber)*.1
        self.thresholdedWeights=append(weights,-threshold)
        Logger.append(0,"initial weights / threashold : " + str(self.thresholdedWeights))
        # adjust neuron
        adjusted=False
        actualLoopNumber=0
        while not adjusted:
            # for each expected result
            adjusted = True # assume neuron is adjusted
            for inputs, expectedOutput in trainings.items():
                Logger.append(1, "expected output : " + str(expectedOutput))
                # compute actual output
                actualOutput=self.activate(inputs)
                Logger.append(1, "actual output : " + str(actualOutput))
                # adjust if needed
                if actualOutput!=expectedOutput:
                    adjusted = False
                    adjustmentSign = expectedOutput - actualOutput
                    Logger.append(1, "adjustment sign : " + str(adjustmentSign))
                    # adjust each activated neuron
                    thresholdedInputs = append(inputs, 1)
                    for neuronIndex, neuronActivated in enumerate(thresholdedInputs):
                        if neuronActivated==1:
                            self.thresholdedWeights[neuronIndex]=self.thresholdedWeights[neuronIndex]+adjustmentSign*adjustmentStep
                    Logger.append(1, "corrected weights / threashold : " + str(self.thresholdedWeights))
            # check loops number
            actualLoopNumber = actualLoopNumber + 1
            if actualLoopNumber >= Neuron.computeLimitLoop:
                message="Sorry, random choices are too long to adjust. Better to retry"
                Logger.append(0, message)
                Logger.flush()
                raise Exception(message)
        # print completed training
        Logger.append(0,"trained in "+str(actualLoopNumber) + " steps :"+linesep+str(self.thresholdedWeights))
        Logger.flush()
        pass
    def activate(self,inputs):
        # sum weighted input
        thresholdedInputs = append(inputs, 1)
        weightedInputs = self.thresholdedWeights.dot(array(thresholdedInputs).transpose())
        # compute & return OUT
        output = heaviside(weightedInputs, 1)
        return output
# train neuron for 'and'
'''trainings={
    ((0, 0)): 0,
    ((0, 1)): 0,
    ((1, 0)): 0,
    ((1, 1)): 1,
}
andNeuron=Neuron(trainings)
print("and 0 0 : " + str(andNeuron.activate((0,0))))
print("and 0 1 : " + str(andNeuron.activate((0,1))))
print("and 1 0 : " + str(andNeuron.activate((1,0))))
print("and 1 1 : " + str(andNeuron.activate((1,1))))'''
# train neuron for 'or'
trainings={
    ((0, 0)): 0,
    ((0, 1)): 1,
    ((1, 0)): 1,
    ((1, 1)): 1,
}
andNeuron=Neuron(trainings)
print("or 0 0 : " + str(andNeuron.activate((0,0))))
print("or 0 1 : " + str(andNeuron.activate((0,1))))
print("or 1 0 : " + str(andNeuron.activate((1,0))))
print("or 1 1 : " + str(andNeuron.activate((1,1))))
