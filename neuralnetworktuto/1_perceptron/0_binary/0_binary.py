#!/usr/bin/env python3
# imports
from matplotlib.pyplot import plot, xticks, yticks, title , xlabel , ylabel, grid, figure, legend, tick_params, savefig
from numpy import heaviside, array, append, arange
from numpy.random import rand
from os import linesep, sep, listdir, makedirs
from os.path import realpath, join, exists
from random import shuffle
from statistics import median, mean
from csv import writer
from shutil import rmtree
# contants
CURRENT_DIRECTORY = realpath(__file__).rsplit(sep, 1)[0]
INPUT_DIRECTORY = join(CURRENT_DIRECTORY,"input")
OUTPUT_DIRECTORY = join(CURRENT_DIRECTORY,"output")
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
def computeDigitStatistics(perceptron, digit,statisticWriter):
    # initialize statistics
    digitWeightsCoalescence={0:list(),1:list()}
    # get digit information
    digitRawInformation=perceptron.digitNeurons[digit].thresholdedWeights
    weights=digitRawInformation[0:-1]
    # for each pixel
    for pixelIndex in range(0, len(weights)):
        # fill statistics details
        pixelValue = perceptron.trainings.data[digit][pixelIndex] # 0 or 1
        digitWeightsCoalescence[pixelValue].append(weights[pixelIndex])
    # write statistics
    writeDigitStatistics(digit, digitWeightsCoalescence, statisticWriter)
    # return
    return digitWeightsCoalescence
def writeDigitStatistics(digit,weightsCoalescence,statisticWriter):
    # compute statistics
    rows=list()
    # for each bit (0,1)
    for bit in weightsCoalescence.keys():
        rows.append(((digit,bit,min(weightsCoalescence[bit]),max(weightsCoalescence[bit]),median(weightsCoalescence[bit]),mean(weightsCoalescence[bit]))))
    comparisonRow=list()
    # compare statistics (1 vs. 0)
    for column in range(2,len(rows[0])):
        comparison="="
        difference=rows[1][column]-rows[0][column]
        if difference>0:
            comparison=">"
        elif difference < 0:
            comparison = "<"
        comparisonRow.append(comparison)
    rows.append(tuple([digit, "1<=>0"]+comparisonRow))
    rows=tuple(rows)
    # write digit statistics
    statisticWriter.writerows(rows)
    # set dedicated figure
    figure()
    # draw weights repartition
    plot(weightsCoalescence[0], "o",color="cyan", label="0")
    plot(weightsCoalescence[1], "o",color="green", label="1")
    tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
    allWeights=tuple(weightsCoalescence[0]+weightsCoalescence[1])
    yticks(arange(round(min(allWeights),1)-.1, round(max(allWeights),1)+.1,.1))
    title("weights repartion for digit : "+str(digit))
    xlabel("pixels")
    ylabel("weight")
    grid(linestyle="-.")
    legend()
    # save figure
    saveFigure("digit#"+str(digit))
def thresholdStatistics(perceptron):
    # coalesce thresholds
    thresholds=list()
    for neuron in perceptron.digitNeurons:
        threshold=-neuron.thresholdedWeights[-1]
        thresholds.append(threshold)
    thresholds = tuple(thresholds)
    # initialize outpout file
    outpoutFile="thresholdsStatistics"
    # write statistics
    statisticReport = open(join(OUTPUT_DIRECTORY,outpoutFile+".csv"), "wt")
    statisticWriter = writer(statisticReport)
    statisticWriter.writerows( (( (("MINIMUM","MAXIMUM","MEDIAN","MEAN")) , ((min(thresholds),max(thresholds),median(thresholds),mean(thresholds) )) )) )
    statisticReport.close()
    # set dedicated figure
    figure()
    # draw weights repartition
    plot(thresholds,"o")
    xticks(arange(0,len(thresholds)+1))
    yticks(arange(round(min(thresholds),1)-.1, round(max(thresholds),1)+.1,.1))
    title("thresholds repartion")
    xlabel("digit")
    ylabel("threshold")
    grid(linestyle="-.")
    # save figure
    saveFigure(outpoutFile)
def saveFigure(name):
    figurePath = join(OUTPUT_DIRECTORY, name + ".png")
    savefig(figurePath)
def main():
    # empty output folder
    if exists(OUTPUT_DIRECTORY):
        rmtree(OUTPUT_DIRECTORY)
    makedirs(OUTPUT_DIRECTORY)
    # train & check neuron network
    images = Images(join(INPUT_DIRECTORY,"training"))
    perceptron = Perceptron(images)
    writeReport(perceptron,perceptron.trainings,join(OUTPUT_DIRECTORY,"trainingReport.txt"))
    # statistic & draw weights/digits graphs ...
    statisticReport = open(join(OUTPUT_DIRECTORY,"digitsStatistics.csv"), "wt")
    statisticWriter = writer(statisticReport)
    statisticWriter.writerow((("DIGIT","BIT","MINIMUM","MAXIMUM","MEDIAN","MEAN")))
    allWeightsCoalescence = {0: list(), 1: list()}
    # for each digits (0 .. 9)
    for digit in range(0,len(perceptron.digitNeurons)):
        digitWeightsCoalescence = computeDigitStatistics(perceptron, digit,statisticWriter)
        # merge weights for global statistics
        for bit in digitWeightsCoalescence.keys():
            allWeightsCoalescence[bit]=allWeightsCoalescence[bit]+digitWeightsCoalescence[bit]
    # write global statistics
    writeDigitStatistics("ALL", allWeightsCoalescence, statisticWriter)
    statisticReport.close()
    # threshold statistics
    thresholdStatistics(perceptron)
    # play with sandbox
    images = Images(join(INPUT_DIRECTORY,"sandbox"))
    writeReport(perceptron,images,join(OUTPUT_DIRECTORY,"sandboxReport.txt"))
class Logger():
    completeLog=""
    @staticmethod
    def append(level, message):
        Logger.completeLog=Logger.completeLog+" "*(4*level)+message+linesep
    @staticmethod
    def flush():
        logFile = open(join(OUTPUT_DIRECTORY,"training.log"),"wt")
        logFile.write(Logger.completeLog)
        logFile.close()
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
    @staticmethod
    def append(errorNumber):
        ErrorsGraph.errorsCounter.append(errorNumber)
    @staticmethod
    def draw():
        # set dedicated figure
        figure()
        # draw training evolution
        plot(ErrorsGraph.errorsCounter, "-o")
        xticks(arange(0, len(ErrorsGraph.errorsCounter) + 1, 1))
        yticks(arange(0, max(ErrorsGraph.errorsCounter) + 1, 1))
        title("training evolution")
        xlabel("training iteration")
        ylabel("errors")
        grid(linestyle="-.", linewidth=.5)
        # save figure
        saveFigure("trainingEvolution")
# digit neuron
class DigitNeuron():
    def __init__(self,digit,retinaLength):
        # set name
        self.digit=digit
        # initialize random weights
        weights=(rand(retinaLength)-.5)*Perceptron.initialCorrectionStep# INFO : we want to balance weights around 0
        threshold=0.125 # INFO : found with a dichotomy between 1 and 0
        self.thresholdedWeights=append(weights,-threshold)
    def activate(self,retinaContext):
        # sum weighted input
        thresholdedInputs = array(append(retinaContext, 1))
        weightedInputs = self.thresholdedWeights.dot(thresholdedInputs.transpose())
        # compute & return OUT
        output = heaviside(weightedInputs, 1)
        return output
    def correct(self,retinaContext,delta):
        # new thresholded weights
        newWeightsThreashold = list()
        # for each pixel on retina
        thresholdedInputs = append(retinaContext, 1)
        for currentIndex,currentValue in enumerate(thresholdedInputs):
            currentWeightThreashold=self.thresholdedWeights[currentIndex]
            # apply correction if needed
            if currentValue==1:
                Logger.append(4,"correction needed -> current pixel or threshold value : " + str(currentValue) + "    current weight : " + str(currentWeightThreashold))
                newWeightThreashold=currentWeightThreashold+delta
                newWeightsThreashold.append(newWeightThreashold)
                Logger.append(4,"new weight : "+str(newWeightThreashold))
            else:
                Logger.append(4,"no correction needed for pixel or threshold value 0")
                newWeightsThreashold.append(currentWeightThreashold)
        # reset neuron weights
        self.thresholdedWeights=array(newWeightsThreashold)
        Logger.append(4,"new neuron : " + str(self))
    def __str__(self):
        representation =str(self.digit) +" : "+str(dict(enumerate(self.thresholdedWeights)))
        return representation
# perceptron
class Perceptron():
    initialCorrectionStep=0.125 # INFO : found with a dichotomy between 1 and 0
    correctionFactor=0.9375 # INFO : found with a dichotomy between 1 and 0.9
    def __init__(self, trainings):
        # set trainings
        self.trainings=trainings
        ErrorsGraph.reset()
        # set number of neurons & neuron input length
        digits = tuple(self.trainings.data.keys())
        digitsNumbers=len(digits)
        retinaLength=len(self.trainings.data[digits[0]])
        # initialize network
        self.initializeDigitNeurons( digitsNumbers, retinaLength)
        Logger.append(0,"digits neurons initialized"+linesep+str(self))
        # assume network is not trained
        trained=False
        # initialize correction step
        self.currentCorrectionStep = Perceptron.initialCorrectionStep
        # initialize training counter
        trainingCounter=0
        # train while necessary
        while not trained:
            Logger.append(0,"training #"+str(trainingCounter)+"   correction step : " + str(self.currentCorrectionStep))
            trainingCounter=trainingCounter+1
            # train all neurons
            trained=self.playAllRandomTrainings()
            # compute next correction step
            self.currentCorrectionStep = self.currentCorrectionStep * Perceptron.correctionFactor
        # print completed training
        Logger.append(0,"trained in "+str(trainingCounter) + " steps :"+linesep+str(self))
        Logger.flush()
        ErrorsGraph.draw()
    def initializeDigitNeurons(self,digitsNumbers,retinaLength):
        # initialize neurons collection
        self.digitNeurons=list()
        # initialize each neurons with random values
        for digit in range(0,digitsNumbers):
            currentDigitNeuron=DigitNeuron(digit,retinaLength)
            self.digitNeurons.append(currentDigitNeuron)
    def playAllRandomTrainings(self):
        # assume network is trained
        trainedPerceptron=True
        errorConter=0
        # shuffle trainings
        shuffledDigits = list(self.trainings.data.keys())
        shuffle(shuffledDigits)
        shuffledDigits=tuple(shuffledDigits)
        Logger.append(1,"training order : "+str(shuffledDigits))
        # for each shuffled training
        for digit in shuffledDigits:
            Logger.append(1,"current training digit : " + str(digit))
            # play current training
            trainedDigit=self.playOneTraining(digit)
            if not trainedDigit:
                errorConter=errorConter+1
                trainedPerceptron=trainedPerceptron and trainedDigit
        # coalesce errors & return
        ErrorsGraph.append(errorConter)
        return trainedPerceptron
    def playOneTraining(self, trainingDigit):
        # assume network is trained
        trained=True
        # compute network outputs
        expectedDigits = [0] * len(self.digitNeurons)
        expectedDigits[trainingDigit] = 1
        expectedDigits = tuple(expectedDigits)
        Logger.append(2,"expected digits : " + str(dict(enumerate(expectedDigits))))
        retinaContext = self.trainings.data[trainingDigit]
        Logger.append(2,"retina context : "+str(trainingDigit)+" -> "+linesep+self.trainings.stringValue(trainingDigit))
        actualDigits = self.execute(retinaContext)
        Logger.append(2,"actual digits : " + str(dict(enumerate(actualDigits))))
        # compare output
        if expectedDigits!=actualDigits:
            Logger.append(2,"this output implies corrections")
            # neuron is not trained
            trained=False
            # check all neurons for correction
            self.checkAllNeuronsCorrection(retinaContext,expectedDigits, actualDigits)
        else:
            Logger.append(2,"this output is fine")
        # return
        return trained
    def execute(self,retinaContext):
        # initialise outputs
        digits=list()
        # compute each neuron output
        for digit in range(0, len(self.digitNeurons)):
            currentOutput = self.digitNeurons[digit].activate(retinaContext)
            digits.append(currentOutput)
        # return
        return tuple(digits)
    def checkAllNeuronsCorrection(self,retinaContext,expectedDigits,actualDigits):
        # for each expected output
        for digit, expectedActivation in enumerate(expectedDigits):
            # get actual output
            actualActivation=actualDigits[digit]
            # check if this neuron need correction
            digitNeuron = self.digitNeurons[digit]
            Logger.append(3,"digit neuron : "+str(digitNeuron))
            if expectedActivation!=actualActivation:
                # compute delta
                delta=self.currentCorrectionStep*(expectedActivation-actualActivation)
                Logger.append(3,"correction delta : "+str(delta))
                # correct this neuron
                digitNeuron.correct(retinaContext,delta)
            else:
                Logger.append(3,"this neuron is fine")
    def __str__(self):
        representation =""
        for digitNeuron in self.digitNeurons:
            representation=representation+str(digitNeuron)+linesep
        return representation
# run script
main()