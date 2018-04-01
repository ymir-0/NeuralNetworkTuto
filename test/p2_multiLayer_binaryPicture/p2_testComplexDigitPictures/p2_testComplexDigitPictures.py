#!/usr/bin/env python3
# imports
from neuralnetworktuto.p1_perceptron.p2_multiLayer_binaryPicture.p2_testComplexPictures import Perceptron, MetaParameters
from os import sep, makedirs
from os.path import join, isfile, realpath, exists
from shutil import rmtree
from json import loads, dumps
# constants
CURRENT_DIRECTORY = realpath(__file__).rsplit(sep, 1)[0]
INPUT_DIRECTORY=join(CURRENT_DIRECTORY,"input")
GLOBAL_TEST_FILE=join(INPUT_DIRECTORY,"TEST.json")
GLOBAL_TRAINING_FILE=join(INPUT_DIRECTORY,"TRAINING.json")
OUTPUT_DIRECTORY = join(CURRENT_DIRECTORY,"output")
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
def testMultipleLayers(layersNumber,loopNumber):
    # create report folder
    reportFolder = join(OUTPUT_DIRECTORY, str(layersNumber) + "layers")
    makedirs(reportFolder)
    # initialize perceptron
    layerHeights = generateLayerHeights(layersNumber)
    perceptron = Perceptron(layerHeights=layerHeights, weightLimit=1, uncertainties=.99)
    # train perceptron
    perceptron.train(TEST_DATA, loopNumber)
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
    testMultipleLayers(4,int(1e4))
    print ("")
    pass
pass
