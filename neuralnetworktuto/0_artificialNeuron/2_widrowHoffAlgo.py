#!/usr/bin/env python3
# imports
from numpy import heaviside, array, append
from numpy.random import rand
# neurons
class Neuron():
    computeLimitLoop=10000 # sometimes, random choices are too long to adjust. better to retry
    def __init__(self,trainings,adjustmentStep):
        # initiate neuron
        weightsNumber=len(tuple(trainings.keys())[0]) # we use the length of 1 expected data key to know weights are required
        weights=rand(weightsNumber)
        threshold=rand(1) # we assume threshold to be an weight so it can be adjust and condidere 'global threshold'=0
        self.thresholdedWeights=append(weights,-threshold)
        # adjust neuron
        adjusted=False
        actualLoopNumber=0
        while not adjusted:
            # for each expected result
            adjusted = True # assume neuron is adjusted
            for inputs, expectedOutput in trainings.items():
                # threshold input
                # compute actual output
                actualOutput=self.activate(inputs)
                # adjust if needed
                if actualOutput!=expectedOutput:
                    adjusted = False
                    if actualOutput<expectedOutput:
                        adjustmentSigne=-1
                    else :
                        adjustmentSigne = 1
                    # adjust each activated neuron
                    thresholdedInputs = append(inputs, 1)
                    for neuronIndex, neuronActivated in enumerate(thresholdedInputs):
                        if neuronActivated==1:
                            self.thresholdedWeights[neuronIndex]=self.thresholdedWeights[neuronIndex]+adjustmentSigne*adjustmentStep
            # check loops number
            actualLoopNumber = actualLoopNumber + 1
            if actualLoopNumber >= Neuron.computeLimitLoop:
                raise ("Sorry, random choices are too long to adjust. Better to retry")
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
andNeuron=Neuron(trainings,0.1)
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
andNeuron=Neuron(trainings,0.1)
print("or 0 0 : " + str(andNeuron.activate((0,0))))
print("or 0 1 : " + str(andNeuron.activate((0,1))))
print("or 1 0 : " + str(andNeuron.activate((1,0))))
print("or 1 1 : " + str(andNeuron.activate((1,1))))
