#!/usr/bin/env python3
# imports
from numpy import heaviside, array, append
from numpy.random import rand
from os import linesep
from random import shuffle
# neuron
class Neuron():
    def __init__(self,name,neuronInputLength):
        # set name
        self.name=name
        # initialize random weights
        weightCoefficient=Perceptron.initialCorrectionStep*(Perceptron.correctionFactor**(41+1)) # INFO : we genraly solve the problem in ~41 steps
        weights=rand(neuronInputLength)*weightCoefficient-(weightCoefficient/2) # INFO : we want to balance weights around 0
        threshold=0.125 # INFO : found with a dichotomy between 1 and 0
        self.thresholdedWeights=append(weights,-threshold)
    def activate(self,input):
        # sum weighted input
        thresholdedInputs = array(append(input, 1))
        weightedInputs = self.thresholdedWeights.dot(thresholdedInputs.transpose())
        # compute & return OUT
        output = heaviside(weightedInputs, 1)
        return output
    def correct(self,input,delta):
        # new thresholded weights
        newThresholdedWeights = list()
        # for each input
        thresholdedInputs = append(input, 1)
        for currentIndex,currentInput in enumerate(thresholdedInputs):
            currentWeight=self.thresholdedWeights[currentIndex]
            # apply correction if needed
            if currentInput==1:
                print("correction needed -> current input : " + str(currentInput) + "    current weight : " + str(currentWeight))
                newWeight=currentWeight+delta
                newThresholdedWeights.append(newWeight)
                print("new weight : "+str(newThresholdedWeights))
                pass
            else:
                print("no correction needed for input value 0")
                newThresholdedWeights.append(currentWeight)
            pass
        # reset neuron weights
        self.thresholdedWeights=array(newThresholdedWeights)
        print("new neurons weights : " + str(self))
    def __str__(self):
        representation =self.name +" : "+str(dict(enumerate(self.thresholdedWeights)))
        return representation
# Perceptron
class Perceptron():
    computeLimitLoop=100 # sometimes, random choices are too long to adjust. better to retry
    initialCorrectionStep=0.125 # INFO : found with a dichotomy between 1 and 0
    correctionFactor=0.9375 # INFO : found with a dichotomy between 1 and 0.9
    def __init__(self, trainings):
        # set trainings
        self.trainings=trainings
        # set number of neurons & neuron input length
        trainingKeys = tuple(self.trainings.data.keys())
        neuronsNumbers=len(self.trainings.data)
        neuronInputLength=len(self.trainings.data[trainingKeys[0]])
        # initialize network
        self.initializeNetwork( neuronsNumbers, neuronInputLength)
        print("neurons initialized"+linesep+str(self))
        # assume network is not trained
        trained=False
        # initialize correction step
        self.currentCorrectionStep = Perceptron.initialCorrectionStep
        # initialize training conter
        trainingCounter=0
        # train while necessary
        while not trained:
            print("training #"+str(trainingCounter)+"   correction step : " + str(self.currentCorrectionStep))
            trainingCounter=trainingCounter+1
            # train all neurons
            trained=self.playAllRandomTrainings()
            # compute next correction step
            self.currentCorrectionStep = self.currentCorrectionStep * Perceptron.correctionFactor
            pass
        # print completed training
        print("TRAINED in "+str(trainingCounter) + " steps :"+linesep+str(self))
    def initializeNetwork(self,neuronsNumbers,neuronInputLength):
        # initialize neurons collection
        self.neurons=list()
        # initialize each neurons with random values
        for neuronIndex in range(0,neuronsNumbers):
            neuronName="neuron#"+str(neuronIndex)
            currentNeuron=Neuron(neuronName,neuronInputLength)
            self.neurons.append(currentNeuron)
        pass
    def playAllRandomTrainings(self):
        # assume network is trained
        trained=True
        # shuffle trainings
        shuffledTrainingKeys = list(self.trainings.data.keys())
        shuffle(shuffledTrainingKeys)
        shuffledTrainingKeys=tuple(shuffledTrainingKeys)
        print("training order : "+str(shuffledTrainingKeys))
        # for each shuffled training
        for currentTrainingKey in shuffledTrainingKeys:
            print("current training value : " + str(currentTrainingKey))
            # play current training
            trained=trained and self.playOneTraining(currentTrainingKey)
            pass
        # return
        return trained
    def playOneTraining(self, trainingKey):
        # assume network is trained
        trained=True
        # compute network outputs
        expectedOutput = [0] * len(self.neurons)
        expectedOutput[trainingKey] = 1
        expectedOutput = tuple(expectedOutput)
        print("expected output : " + str(dict(enumerate(expectedOutput))))
        training = self.trainings.data[trainingKey]
        print("input : "+str(trainingKey)+" -> "+linesep+self.trainings.stringValue(trainingKey))
        actualOutput = self.execute(training)
        print("actual output : " + str(dict(enumerate(actualOutput))))
        # compare output
        if expectedOutput!=actualOutput:
            print("this output implies corrections")
            # neuron is not trained
            trained=False
            # check all neurons for correction
            self.checkAllNeuronsCorrection(training,expectedOutput, actualOutput)
            pass
        else:
            print("this output is fine")
        # return
        return trained
    def execute(self,inputs):
        # initialise outputs
        outputs=list()
        # compute each neuron output
        for neuronIndex in range(0, len(self.neurons)):
            currentOutput = self.neurons[neuronIndex].activate(inputs)
            outputs.append(currentOutput)
        # return
        return tuple(outputs)
    def checkAllNeuronsCorrection(self,input,expectedOutput,actualOutput):
        # for each expected output
        for neuronIndex, neuronExpectedOutput in enumerate(expectedOutput):
            # get actual output
            neuronActualOutput=actualOutput[neuronIndex]
            # check if this neuron need correction
            impliedNeuron = self.neurons[neuronIndex]
            print("implied neuron : "+str(impliedNeuron))
            if neuronExpectedOutput!=neuronActualOutput:
                # compute delta
                delta=self.currentCorrectionStep*(neuronExpectedOutput-neuronActualOutput)
                print("this neuron need corrections delta : "+str(delta))
                # correct this neuron
                impliedNeuron.correct(input,delta)
                pass
            else:
                print("this neuron is fine")
    def __str__(self):
        representation =""
        for currentNeuron in self.neurons:
            representation=representation+str(currentNeuron)+linesep
        return representation
# set training data
class Trainings():
    rowNumber = 6
    columnNumber = 5
    def __init__(self):
        self.data = {
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
                       1, 1, 1, 1, 1]),
            4:
                tuple([0, 1, 0, 0, 0,
                       0, 1, 0, 0, 0,
                       0, 1, 0, 0, 0,
                       0, 1, 0, 1, 0,
                       0, 1, 1, 1, 1,
                       0, 0, 0, 1, 0]),
            5:
                tuple([1, 1, 1, 1, 1,
                       1, 0, 0, 0, 0,
                       1, 0, 0, 0, 0,
                       1, 1, 1, 1, 1,
                       0, 0, 0, 0, 1,
                       1, 1, 1, 1, 1]),
            6:
                tuple([1, 1, 1, 1, 1,
                       1, 0, 0, 0, 0,
                       1, 0, 0, 0, 0,
                       1, 1, 1, 1, 1,
                       1, 0, 0, 0, 1,
                       1, 1, 1, 1, 1]),
            7:
                tuple([1, 1, 1, 1, 0,
                       0, 0, 0, 1, 0,
                       0, 0, 0, 1, 0,
                       0, 0, 1, 1, 1,
                       0, 0, 0, 1, 0,
                       0, 0, 0, 1, 0]),
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
                       1, 1, 1, 1, 1]),
        }
    def stringValue(self,key):
        # initialize representation
        imageRepresentation=""
        # initialize column conter
        columnCounter=0
        # for all pixels in image
        image=self.data[key]
        for pixel in image:
            # set pixel representation
            if pixel==0:
                pixelRepresentation=" "
            else:
                pixelRepresentation = "█"
            # complete image representation
            imageRepresentation=imageRepresentation+pixelRepresentation
            # manage column
            if columnCounter<Trainings.columnNumber-1:
                columnCounter=columnCounter+1
            else:
                imageRepresentation = imageRepresentation +linesep
                columnCounter=0
        # return
        return imageRepresentation
    pass
# tools functions
def prettyStringOutput(output):
    filteredOutput=list()
    for neuronNumber,neuronActivation in enumerate(output):
        if neuronActivation==1:
            filteredOutput.append(neuronNumber)
    return str(tuple(filteredOutput))
# train neuron network
trainings=Trainings()
perceptron=Perceptron(trainings)
# check training
for inputValue, inputImage in trainings.data.items():
    outputValues=perceptron.execute(inputImage)
    outputRepresentation=prettyStringOutput(outputValues)
    print("for image : "+linesep + trainings.stringValue(inputValue)+"corresponding numbers are : " + outputRepresentation)
