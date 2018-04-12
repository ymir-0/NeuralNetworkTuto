#!/usr/bin/env python3
# imports
from neuralnetworktuto.p1_perceptron.p3_stepByStepTraining.p3_stepByStepTraining import Perceptron
from os import sep, makedirs, remove
from os.path import join, isfile, realpath, exists
from shutil import rmtree
from json import loads
# constants
CURRENT_DIRECTORY = realpath(__file__).rsplit(sep, 1)[0]
INPUT_DIRECTORY=join(CURRENT_DIRECTORY,"input")
GLOBAL_TRAINING_FILE=join(INPUT_DIRECTORY,"TRAINING.json")
OUTPUT_DIRECTORY = join(CURRENT_DIRECTORY,"output")
TRAINING_REPORT_NAME = join(OUTPUT_DIRECTORY,"training.csv")
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
def testMultipleLayers(layersNumber,trainingImageNumer):
    # create report folder
    makedirs(OUTPUT_DIRECTORY)
    # initialize perceptron
    layerHeights = generateLayerHeights(layersNumber)
    perceptron = Perceptron(layerHeights=layerHeights, weightLimit=1, uncertainties=.99)
    # train perceptron
    perceptron.trainCompleteSequence(TRAINING_DATA,TRAINING_REPORT_NAME)
    pass
def generateLayerHeights(layersNumber):
    layerHeights =[int(layerHeight * (IMAGE_SIZE-DIGITS_NUMBER) / (layersNumber-1) + DIGITS_NUMBER) for layerHeight in range(0, layersNumber)]
    layerHeights.reverse()
    return tuple(layerHeights)
# define test
if __name__ == "__main__":
    # prepare tests
    TRAINING_DATA = loadGlobalImages(GLOBAL_TRAINING_FILE)
    IMAGE_SIZE = len(TRAINING_DATA[0]["image"])
    DIGITS_NUMBER = 10
    if exists(OUTPUT_DIRECTORY):
        rmtree(OUTPUT_DIRECTORY)
    # train perceptron
    testMultipleLayers(4,5)
    pass
pass
