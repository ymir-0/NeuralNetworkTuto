#!/usr/bin/env python3
# imports
from os import sep
from os.path import join, isfile, realpath
from json import loads
from datetime import datetime
# constants
CURRENT_DIRECTORY = realpath(__file__).rsplit(sep, 1)[0]
GLOBAL_TEST_FILE=join(CURRENT_DIRECTORY,"TEST.json")
GLOBAL_TRAINING_FILE=join(CURRENT_DIRECTORY,"TRAINING.json")
# utility methods
def loadGlobalImages(file):
    # read global file
    dataFile = open(file)
    globalContent = dataFile.read()
    dataFile.close()
    # load data
    globalData=loads(globalContent)
    # return
    return globalData
    pass
pass
# define test
if __name__ == "__main__":
    # load training data
    print ("start : " + str(datetime.now()))
    testData = loadGlobalImages(GLOBAL_TEST_FILE)
    trainingData = loadGlobalImages(GLOBAL_TRAINING_FILE)
    print ("end : " + str(datetime.now()))
    pass
pass
