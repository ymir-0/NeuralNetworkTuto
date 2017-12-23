#!/usr/bin/env python3
# imports
from numpy import heaviside, array, append
from numpy.random import rand
from os import linesep
# neuron
class Neuron():
    def __init__(self,name,neuronInputLength):
        # set name
        self.name=name
        # initialize random weights
        weights=rand(neuronInputLength) # TODO: explain magic number
        threshold=rand(1) # we assume threshold to be an weight so it can be adjust and condidere 'global threshold'=0
        self.thresholdedWeights=tuple(append(weights,-threshold))
    def activate(self,inputs):
        # sum weighted input
        thresholdedInputs = append(inputs, 1)
        weightedInputs = self.thresholdedWeights.dot(array(thresholdedInputs).transpose())
        # compute & return OUT
        output = heaviside(weightedInputs, 1)
        return output
    def __str__(self):
        representation =self.name +" : "+str(dict(enumerate(self.thresholdedWeights)))
        return representation
class Perceptron():
    computeLimitLoop=10000 # sometimes, random choices are too long to adjust. better to retry
    initialCorrectionStep=1 # TODO: explain magic number
    correctionFactor=0.75 # TODO: explain magic number
    def __init__(self, trainings):
        # set number of neurons & neuron input length
        trainingKeys = tuple(trainings.keys())
        neuronsNumbers=len(trainings)
        neuronInputLength=len(trainings[trainingKeys[0]])
        # initialize neurons
        self.initializeNeurons( neuronsNumbers, neuronInputLength)
        print("Neurons initialized"+linesep+str(self))
        pass
    def initializeNeurons(self,neuronsNumbers,neuronInputLength):
        # initialize neurons collection
        self.neurons=list()
        # initialize each neurons with random values
        for neuronIndex in range(0,neuronsNumbers):
            neuronName="neuron#"+str(neuronIndex)
            self.neurons.append(Neuron(neuronName,neuronInputLength))
        pass
    def __str__(self):
        representation =""
        for currentNeuron in self.neurons:
            representation=representation+str(currentNeuron)+linesep
        return representation
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
    #1: completeTrainings[1],
    #2: completeTrainings[2],
    #3: completeTrainings[3],
    #4: completeTrainings[4],
    #5: completeTrainings[5],
    #6: completeTrainings[6],
    #7: completeTrainings[7],
    #8: completeTrainings[8],
    #9: completeTrainings[9],
}
# train neuron network
perceptron=Perceptron(trainings)
# test results
print("N ->  0    1    2    3    4    5")
#for inputValue, inputImage in trainings.items():
#    outputValues=perceptron.execute(inputImage)
#    print(str(inputValue) + " -> " + str(outputValues))
