#!/usr/bin/env python3
# imports
from matplotlib.pyplot import plot, xticks, yticks, title , xlabel , ylabel, grid, figure, legend, tick_params, savefig
from numpy import array, append, arange
from numpy.random import rand
from os import linesep, sep, listdir, makedirs
from os.path import realpath, join, exists
from random import shuffle
from statistics import median, mean, pstdev
from csv import writer
from shutil import rmtree
from math import exp, log
# contants
CURRENT_DIRECTORY = realpath(__file__).rsplit(sep, 1)[0]
INPUT_DIRECTORY = join(CURRENT_DIRECTORY,"input")
OUTPUT_DIRECTORY = join(CURRENT_DIRECTORY,"output")
INITIAL_UNCERTAINTY = .1 # initial uncertainty percentage
UNCERTAINTY_TEST_NUMBER = int(1e3)
# tools functions
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
def optimizeNeuronUncertainty(perceptron,digit):
    neuron=perceptron.digitNeurons[digit]
    training=perceptron.trainings.data[digit]
    # optimize uncertainty
    neuron.uncertainty = - neuron.weightedInputs(training) / log ( INITIAL_UNCERTAINTY / (100-INITIAL_UNCERTAINTY) )
    pass
def thresholdStatistics(perceptron):
    # coalesce thresholds & uncertainties
    thresholds=list()
    uncertainties = list()
    for neuron in perceptron.digitNeurons:
        # thresholds
        threshold=-neuron.thresholdedWeights[-1]
        thresholds.append(threshold)
        # uncertainties
        uncertainties.append(neuron.uncertainty)
    thresholds = tuple(thresholds)
    uncertainties = tuple(uncertainties)
    # write statistics
    statisticReport = open(join(OUTPUT_DIRECTORY,"thresholdsUncertaintiesStatistics"+".csv"), "wt")
    statisticWriter = writer(statisticReport)
    statisticWriter.writerows( ((
        (("DATA","MINIMUM","MAXIMUM","MEDIAN","MEAN","DEVIATION")) ,
        (("THREASHOLDS", min(thresholds),max(thresholds),median(thresholds),mean(thresholds),pstdev(thresholds) )),
        (("UNCERTAINTIES", min(uncertainties), max(uncertainties), median(uncertainties), mean(uncertainties),pstdev(uncertainties) )),
    )) )
    statisticReport.close()
    # draw weights & uncertainties repartition
    figure()
    plot(thresholds,"o", label="weights")
    plot(uncertainties,"x", label="uncertainties")
    xticks(arange(0,len(perceptron.digitNeurons)+1))
    title("thresholds & uncertainties repartition (for "+str(INITIAL_UNCERTAINTY)+"% success on training)")
    xlabel("digit")
    ylabel("threshold & uncertainties")
    grid(linestyle="-.")
    legend()
    saveFigure("thresholdsUncertaintiesRepartition")
    # draw uncertainty in terms of threshold
    figure()
    plot(thresholds, uncertainties,"o")
    title("uncertainty in terms of thresholds")
    xlabel("threshold")
    ylabel("uncertainty")
    grid(linestyle="-.")
    saveFigure("thresholdUncertaintyCorrelation")
    pass
def drawErrorsEvolution(perceptron):
    # initialize
    totalPixelsNumber=Images.rowNumber*Images.columnNumber
    globalErrorsCounter=list()
    digitsNumber=len(perceptron.digitNeurons)
    # for each digit
    for digit, neuron in enumerate(perceptron.digitNeurons):
        image=perceptron.trainings.data[digit]
        digitErrorsCounter = list()
        # switch all pixels step by step
        for pixelSwitchNumber in range(0,totalPixelsNumber):
            errorCounter=0
            # test a lot of time
            for testNumber in range(0,UNCERTAINTY_TEST_NUMBER):
                print("TEST digit:"+str(digit)+"/"+str(digitsNumber-1)+"\tpixel swith:"+str(pixelSwitchNumber)+"/"+str(totalPixelsNumber-1)+"\ttest:"+str(testNumber)+"/"+str(UNCERTAINTY_TEST_NUMBER-1))
                corruptedImage = swithImagePixel(image, pixelSwitchNumber)
                # sometimes, we have a math error
                try:
                    # track each error
                    activated=neuron.activate(corruptedImage)
                    if not activated:
                        errorCounter=errorCounter+1
                # retry
                except:
                    pixelSwitchNumber=pixelSwitchNumber-1
                pass
            # coalesce digit errors
            relativeError = errorCounter / UNCERTAINTY_TEST_NUMBER * 100
            digitErrorsCounter.append(relativeError)
            pass
        # draw error evolution
        digitErrorsCounter = tuple(digitErrorsCounter)
        relatedAmortizedParameter=amortizedParameter(digitErrorsCounter)
        inflexion, relatedSigmoidParameter=sigmoidParameters(digitErrorsCounter)
        relatedLinearParameter =linearParameters(digitErrorsCounter)
        figure()
        plot(digitErrorsCounter, "-o", label="error evolution")
        absciseRange=range(0,30)
        plot(absciseRange,[100*(1-exp(-x/relatedAmortizedParameter)) for x in absciseRange], label="armortized curve")
        plot(absciseRange,[100/(1+exp(relatedSigmoidParameter*(inflexion-x))) for x in absciseRange], label="logistic curve")
        plot(absciseRange,[relatedLinearParameter*x+INITIAL_UNCERTAINTY for x in absciseRange], label="linear curve")
        title("error evolution for digit "+str(digit))
        xlabel("number of swtiched pixels")
        ylabel("relative error %")
        grid(linestyle="-.")
        legend()
        saveFigure("errorEvolutionDigit#"+str(digit))
        # coalesce all errors
        globalErrorsCounter.append(digitErrorsCounter)
        pass
    # draw all errors
    globalErrorsCounter = tuple(globalErrorsCounter)
    figure()
    for digit, errorEvolution in enumerate(globalErrorsCounter):
        plot(errorEvolution, "-o",label=str(digit))
    title("error evolution for all digits ")
    xlabel("number of swtiched pixels")
    ylabel("relative error %")
    grid(linestyle="-.")
    legend()
    saveFigure("errorEvolutionDigitAll")
    pass
def amortizedParameter(digitErrorsCounter):
    # compute all possible parameters value
    # WARNING : no solution at 0
    parameters=list()
    for errorIndex in range(1, len(digitErrorsCounter)):
        # sometimes, we have a math error
        try:
            # track each error
            candidateParameter=-errorIndex/log(1-digitErrorsCounter[errorIndex]/100)
            parameters.append(candidateParameter)
        except: pass
        pass
    parameter=mean(parameters)
    return parameter
def sigmoidParameters(digitErrorsCounter):
    # INFO : logistic function is increasing, so we use a dichotomy to det inflexion point
    # initialize inflexion search
    x = 0
    sign=1
    xStep=round(len(digitErrorsCounter)/2)
    # while we did not surrend inflexion
    while xStep > 1:
        # compute curent surrounding value
        x=x+sign*xStep
        y=digitErrorsCounter[x]
        # if (by any chance) we get the exact inflexion point, return it as it is
        if y==50 :
            inflexion=x
            break
        # compute next surrounding value
        else :
            sign=1 if y<50 else -1
            xStep=round(xStep/2)
            # if surrending is done
            if xStep==1:
                # get previous surrounding point
                xp=x
                yp=y
                # get next surrounding point
                x=x+sign
                y=digitErrorsCounter[x]
                # compute line throw surrounding points
                A=(yp-y)/(xp-x)
                B=(yp*x-y*xp)/(x-xp)
                # compute inflexion point
                inflexion = (50-B)/A
                pass
    # compute all possible parameters value
    parameters=list()
    for errorIndex in range(0, len(digitErrorsCounter)):
        # sometimes, we have a math error
        try:
            # track each error
            candidateParameter = log(100/digitErrorsCounter[errorIndex]-1) / (inflexion - errorIndex)
            parameters.append(candidateParameter)
        except: pass
        pass
    parameter=mean(parameters)
    return inflexion, parameter
def linearParameters(digitErrorsCounter):
    # compute all possible parameters value
    parameters=list()
    for errorIndex in range(1, len(digitErrorsCounter)):
        # sometimes, we have a math error
        try:
            # track each error
            candidateParameter = (digitErrorsCounter[errorIndex] - digitErrorsCounter[1]) / (errorIndex-1)
            parameters.append(candidateParameter)
        except: pass
        pass
    parameter=mean(parameters)
    return parameter
def swithImagePixel(originalImage, pixelSwitchNumber):
    # shuffle pixel indexes
    shuffledPixelsIndexes=list(range(0,len(originalImage)))
    shuffle(shuffledPixelsIndexes)
    shuffledPixelsIndexes=shuffledPixelsIndexes[0:pixelSwitchNumber]
    # switch pixels
    newImage = list(originalImage)
    for pixelIndex in shuffledPixelsIndexes:
        newImage[pixelIndex]=1-originalImage[pixelIndex]
        pass
    # return
    return tuple(newImage)
def saveFigure(name):
    figurePath = join(OUTPUT_DIRECTORY, name + ".png")
    savefig(figurePath)
def main():
    # empty output folder
    if exists(OUTPUT_DIRECTORY):
        rmtree(OUTPUT_DIRECTORY)
    makedirs(OUTPUT_DIRECTORY)
    # train neuron network
    images = Images(join(INPUT_DIRECTORY,"training"))
    perceptron = Perceptron(images)
    # statistic & draw weights/digits graphs ...
    statisticReport = open(join(OUTPUT_DIRECTORY,"digitsStatistics.csv"), "wt")
    statisticWriter = writer(statisticReport)
    statisticWriter.writerow((("DIGIT","BIT","MINIMUM","MAXIMUM","MEDIAN","MEAN")))
    allWeightsCoalescence = {0: list(), 1: list()}
    # for each digits (0 .. 9)
    for digit in range(0,len(perceptron.digitNeurons)):
        # compute digit statistics
        digitWeightsCoalescence = computeDigitStatistics(perceptron, digit,statisticWriter)
        # merge weights for global statistics
        for bit in digitWeightsCoalescence.keys():
            allWeightsCoalescence[bit]=allWeightsCoalescence[bit]+digitWeightsCoalescence[bit]
        # optimize uncertainty
        optimizeNeuronUncertainty(perceptron,digit)
    # complete logs
    Logger.append(0, "optimized uncertainties" + linesep + str(perceptron))
    Logger.flush()
    # write global statistics
    writeDigitStatistics("ALL", allWeightsCoalescence, statisticWriter)
    statisticReport.close()
    # threshold statistics
    thresholdStatistics(perceptron)
    # test digits
    drawErrorsEvolution(perceptron)
# tools class
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
    def __init__(self,digit,retinaLength,uncertainty=1e-2): # INFO : 1e-2 is a fine uncertainty to approch binary value without floating error
        # set parameters
        self.digit=digit
        self.uncertainty=uncertainty
        # initialize random weights
        initialWeights=(rand(retinaLength)-.5)*Perceptron.initialCorrectionStep# INFO : we want to balance weights around 0
        self.thresholdedWeights=append(initialWeights,-Perceptron.initialCorrectionStep)
    def activate(self,retinaContext):
        # sum weighted input
        weightedInputs = self.weightedInputs(retinaContext)
        # compute probabilistic activation
        thresholdProbability = 1 / (1 + exp(-weightedInputs/self.uncertainty))
        activationRandomChoice=rand()
        output= activationRandomChoice <= thresholdProbability
        # return OUT
        return output
    def weightedInputs(self,retinaContext):
        thresholdedInputs = array(append(retinaContext, 1))
        weightedInputs = self.thresholdedWeights.dot(thresholdedInputs.transpose())
        return weightedInputs
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
        representation =str(self.digit) +" : [ threashold : " + str(self.thresholdedWeights[-1]) + " ; uncertainty : " + str(self.uncertainty) + " ; weights : " + str(dict(enumerate(self.thresholdedWeights[0:-1]))) + " ]"
        return representation
# perceptron
class Perceptron():
    computeLimitLoop=30 # sometimes, random choices are too long to adjust. better to retry
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
        actualLoopNumber=0
        while not (trained or trainingCounter == Perceptron.computeLimitLoop):
            Logger.append(0,"training #"+str(trainingCounter)+"   correction step : " + str(self.currentCorrectionStep))
            trainingCounter=trainingCounter+1
            # train all neurons
            trained=self.playAllRandomTrainings()
            # compute next correction step
            self.currentCorrectionStep = self.currentCorrectionStep * Perceptron.correctionFactor
            if trainingCounter >= Perceptron.computeLimitLoop:
                message = "Sorry, random choices are too long to adjust. Better to retry"
                Logger.append(0, message)
                Logger.flush()
                #raise Exception(message)
        # print completed training
        Logger.append(0,"trained in "+str(trainingCounter) + " steps :"+linesep+str(self))
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