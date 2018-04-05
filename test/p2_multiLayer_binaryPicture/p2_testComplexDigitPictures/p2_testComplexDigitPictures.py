#!/usr/bin/env python3
# imports
from neuralnetworktuto.p1_perceptron.p2_multiLayer_binaryPicture.p2_testComplexPictures import Perceptron
from os import sep, makedirs, remove
from os.path import join, isfile, realpath, exists
from shutil import rmtree
from json import loads
from csv import writer
import csv
# constants
CURRENT_DIRECTORY = realpath(__file__).rsplit(sep, 1)[0]
INPUT_DIRECTORY=join(CURRENT_DIRECTORY,"input")
GLOBAL_TRAINING_FILE=join(INPUT_DIRECTORY,"TRAINING.json")
GLOBAL_TEST_FILE=join(INPUT_DIRECTORY,"TEST.json")
OUTPUT_DIRECTORY = join(CURRENT_DIRECTORY,"output")
TRAINING_LOG = join(OUTPUT_DIRECTORY,"training.log")
TEST_LOG = join(OUTPUT_DIRECTORY,"test.log")
TEST_MEASURES = join(OUTPUT_DIRECTORY,"test.csv")
# utility methods
def loadGlobalImages(file):
    # read global file
    dataFile = open(file)
    globalContent = dataFile.read()
    dataFile.close()
    # load data
    globalData=loads(globalContent)
    # return
    return tuple(globalData)
    pass
pass
def testPerceptron(perceptron):
    # initialize & open log file
    if exists(TEST_LOG):
        remove(TEST_LOG)
    testLogFile = open(TEST_LOG, 'a')
    # initialize measures
    header=["EXPECTED_DIGIT"
        ,"DIGIT_0_POTENTIAL"
        ,"DIGIT_1_POTENTIAL"
        ,"DIGIT_2_POTENTIAL"
        ,"DIGIT_3_POTENTIAL"
        ,"DIGIT_4_POTENTIAL"
        ,"DIGIT_5_POTENTIAL"
        ,"DIGIT_6_POTENTIAL"
        ,"DIGIT_7_POTENTIAL"
        ,"DIGIT_8_POTENTIAL"
        ,"DIGIT_9_POTENTIAL"
        ,"MOST_POTENTIAL_DIGIT"
        ,"EXPECTED_POTENTIAL_DIGIT"
        ]
    measures=list()
    measures.append(header)
    # test each image
    for testData in TEST_DATA:
        # parse test data
        image=testData["image"]
        expectedDigitVector=testData["label"]
        # compute image test result
        expectedDigit=expectedDigitVector.index(1)
        actualDigitVector=perceptron.passForward(image)
        actualMostPotentialDigit=actualDigitVector.index(max(actualDigitVector))
        expectedDigitPotential=actualDigitVector[expectedDigit]
        # print log
        testLogFile.write(
            "expectedDigit : " + str(expectedDigit) +
            " | actualDigitVector : " + str(actualDigitVector) +
            " | actualMostPotentialDigit : " + str(actualMostPotentialDigit) +
            " | expectedDigitPotential : " + str(expectedDigitPotential) +
            "\n")
        # store measure
        measure=tuple([expectedDigit]+list(actualDigitVector)+[actualMostPotentialDigit,expectedDigitPotential])
        measures.append(measure)
        pass
    # close log file
    testLogFile.close()
    # write measures
    with open(TEST_MEASURES, 'w') as testMeasuresReport:
        reportWriter = writer(testMeasuresReport, delimiter=';')
        reportWriter.writerows(measures)
    pass
def testMultipleLayers(layersNumber,maximumTime):
    # create report folder
    makedirs(OUTPUT_DIRECTORY)
    # initialize perceptron
    layerHeights = generateLayerHeights(layersNumber)
    perceptron = Perceptron(layerHeights=layerHeights, weightLimit=1, uncertainties=.99)
    # train perceptron
    perceptron.train(TRAINING_DATA,TRAINING_LOG, maximumTime=maximumTime)
    # test perceptron
    testPerceptron(perceptron)
    pass
def generateLayerHeights(layersNumber):
    layerHeights =[int(layerHeight * (IMAGE_SIZE-DIGITS_NUMBER) / (layersNumber-1) + DIGITS_NUMBER) for layerHeight in range(0, layersNumber)]
    layerHeights.reverse()
    return tuple(layerHeights)
# define test
if __name__ == "__main__":
    # prepare tests
    TEST_DATA = loadGlobalImages(GLOBAL_TEST_FILE)
    TRAINING_DATA = loadGlobalImages(GLOBAL_TRAINING_FILE)
    IMAGE_SIZE = len(TEST_DATA[0]["image"])
    DIGITS_NUMBER = 10
    if exists(OUTPUT_DIRECTORY):
        rmtree(OUTPUT_DIRECTORY)
    # train perceptron
    testMultipleLayers(4,1*60)
    print ("")
    pass
pass
