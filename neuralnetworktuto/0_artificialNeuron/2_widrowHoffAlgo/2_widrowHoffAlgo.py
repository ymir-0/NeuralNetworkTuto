#!/usr/bin/env python3
# imports
from numpy import heaviside, array, append
from numpy.random import rand
from os import linesep, sep, makedirs
from os.path import join, realpath, exists
from shutil import rmtree
# contants
CURRENT_DIRECTORY = realpath(__file__).rsplit(sep, 1)[0]
OUTPUT_DIRECTORY = join(CURRENT_DIRECTORY,"output")
# tools class
class Logger():
    completeLog=""
    @staticmethod
    def append(level, message):
        Logger.completeLog=Logger.completeLog+" "*(4*level)+message+linesep
    @staticmethod
    def flush():
        logFile = open(join(OUTPUT_DIRECTORY,"training_" + Logger.name + ".log"),"wt")
        logFile.write(Logger.completeLog)
        logFile.close()
# neurons
class Neuron():
    computeLimitLoop=300 # sometimes, random choices are too long to adjust. better to retry
    adjustmentStep = .01
    def __init__(self,trainings):
        # initiate neuron
        weightsNumber=len(tuple(trainings.keys())[0]) # we use the length of 1st training data to get neurons number
        # INFO : found with a dichotomy between 1 and 0 for doing some loops
        threshold=rand()*.6+.1
        weights=rand(weightsNumber)*.1-.2
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
                            self.thresholdedWeights[neuronIndex]=self.thresholdedWeights[neuronIndex]+adjustmentSign*Neuron.adjustmentStep
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
# empty output folder
if exists(OUTPUT_DIRECTORY):
    rmtree(OUTPUT_DIRECTORY)
makedirs(OUTPUT_DIRECTORY)
# train & test neuron for 'and'
operatorsTrainings={
    "AND" : {
        ((0, 0)): 0,
        ((0, 1)): 0,
        ((1, 0)): 0,
        ((1, 1)): 1,
    },
    "OR": {
        ((0, 0)): 0,
        ((0, 1)): 1,
        ((1, 0)): 1,
        ((1, 1)): 1,
    },
}
for operator, trainings in operatorsTrainings.items():
    Logger.name=operator
    report = ""
    neuron = Neuron(trainings)
    for A in range (0,2):
        for B in range(0, 2):
            report = report + str(A) + " , " + str(B) + " : " + str(neuron.activate((A, B))) + linesep
    reportFile = open(join(OUTPUT_DIRECTORY, "test_" + operator + ".txt"), "wt")
    reportFile.write(report)
    reportFile.close()
