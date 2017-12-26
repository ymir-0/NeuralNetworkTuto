#!/usr/bin/env python3
# imports
from matplotlib.pyplot import plot, show, xticks, yticks, title , xlabel , ylabel, grid
from numpy import heaviside, array, append, arange
from numpy.random import rand
from os import linesep, sep, listdir
from os.path import realpath, join
from random import shuffle
# contants
CURRENT_DIRECTORY = realpath(__file__).rsplit(sep, 1)[0]
TRAINING_FOLDER=join(CURRENT_DIRECTORY,"input","training")
SANDBOX_FOLDER=join(CURRENT_DIRECTORY,"input","sandbox")
TRAINING_LOG=join(CURRENT_DIRECTORY,"output","training.log")
TRAINING_REPORT=join(CURRENT_DIRECTORY,"output","trainingReport.txt")
SANDBOX_REPORT=join(CURRENT_DIRECTORY,"output","sandboxReport.txt")
# tools functions
def prettyStringOutput(output):
    filteredOutput=list()
    for neuronNumber,neuronActivation in enumerate(output):
        if neuronActivation==1:
            filteredOutput.append(neuronNumber)
    return str(tuple(filteredOutput))
def writeReport(perceptron,images,reportFileName):
    # initialize training report
    trainingReport = ""
    # for all training value
    for inputValue, inputImage in images.data.items():
        # fill report
        output = perceptron.execute(inputImage)
        outputRepresentation = prettyStringOutput(output)
        trainingReport = trainingReport+"for image : " + linesep + images.stringValue(inputValue) + "corresponding numbers are : " + outputRepresentation + linesep
        # write report
        reportFile = open(reportFileName,"wt")
        reportFile.write(trainingReport)
        reportFile.close()
        pass
    pass
def main():
    # train & check neuron network
    images = Images(TRAINING_FOLDER)
    perceptron = Perceptron(images)
    writeReport(perceptron,perceptron.trainings,TRAINING_REPORT)
    # play with sandbox
    images = Images(SANDBOX_FOLDER)
    writeReport(perceptron,images,SANDBOX_REPORT)
    pass
# tools classes
class Logger():
    completeLog=""
    @staticmethod
    def append(level, message):
        Logger.completeLog=Logger.completeLog+" "*(4*level)+message+linesep
        pass
    @staticmethod
    def flush():
        logFile = open(TRAINING_LOG,"wt")
        logFile.write(Logger.completeLog)
        logFile.close()
    pass
class Images():
    rowNumber = 6
    columnNumber = 5
    def __init__(self,dataFolder):
        # initialize data
        self.data=dict()
        # for each data file
        for dataFileShortName in listdir(dataFolder):
            # extract data key
            key=int(dataFileShortName.split(".")[0])
            # read it
            dataFileFullName=join(dataFolder,dataFileShortName)
            dataFile = open(dataFileFullName)
            rawData=dataFile.read()
            dataFile.close()
            # construct image
            dataPivot=rawData.replace(linesep,"")
            image=list()
            for pixel in dataPivot:
                image.append(int(pixel))
                pass
            # fill data
            self.data[key]=tuple(image)
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
            if columnCounter<Images.columnNumber-1:
                columnCounter=columnCounter+1
            else:
                imageRepresentation = imageRepresentation +linesep
                columnCounter=0
        # return
        return imageRepresentation
class ErrorsGraph():
    errorsCounter=list()
    @staticmethod
    def reset():
        ErrorsGraph.errorsCounter=list()
        pass
    @staticmethod
    def append(errorNumber):
        ErrorsGraph.errorsCounter.append(errorNumber)
        pass
    @staticmethod
    def draw():
        plot(ErrorsGraph.errorsCounter, "-o")
        xticks(arange(0, len(ErrorsGraph.errorsCounter) + 1, 1))
        yticks(arange(0, max(ErrorsGraph.errorsCounter) + 1, 1))
        title("training evolution")
        xlabel("training loop")
        ylabel("errors")
        grid(linestyle="-.", linewidth=.5)
        show()
        pass
    pass
pass
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
                Logger.append(4,"correction needed -> current input : " + str(currentInput) + "    current weight : " + str(currentWeight))
                newWeight=currentWeight+delta
                newThresholdedWeights.append(newWeight)
                Logger.append(4,"new weight : "+str(newWeight))
                pass
            else:
                Logger.append(4,"no correction needed for input value 0")
                newThresholdedWeights.append(currentWeight)
            pass
        # reset neuron weights
        self.thresholdedWeights=array(newThresholdedWeights)
        Logger.append(4,"new neurons weights : " + str(self))
    def __str__(self):
        representation =self.name +" : "+str(dict(enumerate(self.thresholdedWeights)))
        return representation
# perceptron
class Perceptron():
    computeLimitLoop=100 # sometimes, random choices are too long to adjust. better to retry
    initialCorrectionStep=0.125 # INFO : found with a dichotomy between 1 and 0
    correctionFactor=0.9375 # INFO : found with a dichotomy between 1 and 0.9
    def __init__(self, trainings):
        # set trainings
        self.trainings=trainings
        ErrorsGraph.reset()
        # set number of neurons & neuron input length
        trainingKeys = tuple(self.trainings.data.keys())
        neuronsNumbers=len(self.trainings.data)
        neuronInputLength=len(self.trainings.data[trainingKeys[0]])
        # initialize network
        self.initializeNetwork( neuronsNumbers, neuronInputLength)
        Logger.append(0,"neurons initialized"+linesep+str(self))
        # assume network is not trained
        trained=False
        # initialize correction step
        self.currentCorrectionStep = Perceptron.initialCorrectionStep
        # initialize training conter
        trainingCounter=0
        # train while necessary
        while not trained:
            Logger.append(0,"training #"+str(trainingCounter)+"   correction step : " + str(self.currentCorrectionStep))
            trainingCounter=trainingCounter+1
            # train all neurons
            trained=self.playAllRandomTrainings()
            # compute next correction step
            self.currentCorrectionStep = self.currentCorrectionStep * Perceptron.correctionFactor
            pass
        # print completed training
        Logger.append(0,"TRAINED in "+str(trainingCounter) + " steps :"+linesep+str(self))
        Logger.flush()
        ErrorsGraph.draw()
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
        errorConter=0
        # shuffle trainings
        shuffledTrainingKeys = list(self.trainings.data.keys())
        shuffle(shuffledTrainingKeys)
        shuffledTrainingKeys=tuple(shuffledTrainingKeys)
        Logger.append(1,"training order : "+str(shuffledTrainingKeys))
        # for each shuffled training
        for currentTrainingKey in shuffledTrainingKeys:
            Logger.append(1,"current training value : " + str(currentTrainingKey))
            # play current training
            currentTrained=self.playOneTraining(currentTrainingKey)
            if not currentTrained:
                errorConter=errorConter+1
            trained=trained and currentTrained
            pass
        # coalesce errors & return
        ErrorsGraph.append(errorConter)
        return trained
    def playOneTraining(self, trainingKey):
        # assume network is trained
        trained=True
        # compute network outputs
        expectedOutput = [0] * len(self.neurons)
        expectedOutput[trainingKey] = 1
        expectedOutput = tuple(expectedOutput)
        Logger.append(2,"expected output : " + str(dict(enumerate(expectedOutput))))
        training = self.trainings.data[trainingKey]
        Logger.append(2,"input : "+str(trainingKey)+" -> "+linesep+self.trainings.stringValue(trainingKey))
        actualOutput = self.execute(training)
        Logger.append(2,"actual output : " + str(dict(enumerate(actualOutput))))
        # compare output
        if expectedOutput!=actualOutput:
            Logger.append(2,"this output implies corrections")
            # neuron is not trained
            trained=False
            # check all neurons for correction
            self.checkAllNeuronsCorrection(training,expectedOutput, actualOutput)
            pass
        else:
            Logger.append(2,"this output is fine")
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
            Logger.append(3,"implied neuron : "+str(impliedNeuron))
            if neuronExpectedOutput!=neuronActualOutput:
                # compute delta
                delta=self.currentCorrectionStep*(neuronExpectedOutput-neuronActualOutput)
                Logger.append(3,"this neuron need corrections delta : "+str(delta))
                # correct this neuron
                impliedNeuron.correct(input,delta)
                pass
            else:
                Logger.append(3,"this neuron is fine")
    def __str__(self):
        representation =""
        for currentNeuron in self.neurons:
            representation=representation+str(currentNeuron)+linesep
        return representation
# run script
main()