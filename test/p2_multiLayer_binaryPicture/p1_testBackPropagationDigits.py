#!/usr/bin/env python3
# PY test script file name must start with "test" to allow automatic recognition by PyCharm
# import
import unittest
from neuralnetworktuto.p1_perceptron.p2_multiLayer_binaryPicture.p1_testBackPropagation import Perceptron, MetaParameters
from numpy import exp, newaxis, zeros, array, sum
from numpy.ma import size
from numpy.random import rand
from os import linesep, sep, listdir, makedirs
from os.path import realpath, join, exists
from random import shuffle
from shutil import rmtree
from collections import Iterable
from enum import Enum, unique
from copy import deepcopy
from statistics import mean
from matplotlib.pyplot import plot, xticks, yticks, title , xlabel , ylabel, grid, figure, legend, tick_params, savefig, show
# contants
CURRENT_DIRECTORY = realpath(__file__).rsplit(sep, 1)[0]
INPUT_DIRECTORY = join(CURRENT_DIRECTORY,"input")
# utility methods
def readTest():
    # initialize data
    sequences = dict()
    # for each data file
    for dataFileShortName in listdir(INPUT_DIRECTORY):
        # extract data key
        expectedDigit = int(dataFileShortName.split(".")[0])
        digits = [0]*10
        digits[expectedDigit] = 1
        digits = tuple(digits)
        # read it
        dataFileFullName = join(INPUT_DIRECTORY, dataFileShortName)
        dataFile = open(dataFileFullName)
        rawData = dataFile.read()
        dataFile.close()
        # construct image
        dataPivot = rawData.replace(linesep, "")
        image = list()
        for pixel in dataPivot:
            image.append(int(pixel))
        image = tuple(image)
        # fill data
        sequences[image] = digits
    # return
    return sequences
SEQUENCES = readTest()
def generateLayerHeights(layersNumber):
    layerHeights =[int(layerHeight * 20 / (layersNumber-1) + 10) for layerHeight in range(0, layersNumber)]
    layerHeights.reverse()
    return tuple(layerHeights)
def trainPerceptron(name, perceptron, loopNumber, metaParametersUpdate=((MetaParameters.BIASES.value))):
    # train perceptron
    print("test : " + name)
    print("loop number = " + str(loopNumber))
    errors = perceptron.train(SEQUENCES, loopNumber, metaParametersUpdate=metaParametersUpdate)
    # display results
    for input, expectedOutput in SEQUENCES.items():
        # print results
        actualOutput = perceptron.passForward(input)
        print("input = " + str(input) + "\texpected output = " + str(expectedOutput) + "\tactual output = " + str(
            actualOutput) + "\terror = " + str(errors[expectedOutput][-1]))
        # prepare graph
        plot(errors["loopNumbers"], errors[expectedOutput], label=str(expectedOutput))
    title("errors evolution (test : " + name + ")")
    xlabel("training step")
    ylabel("error")
    grid(linestyle="-.")
    legend()
    show()
    pass
def testMultipleLayers(layersNumber,loopNumber):
    # initialize perceptron
    layerHeights = generateLayerHeights(layersNumber)
    perceptron = Perceptron(layerHeights=layerHeights, weightLimit=1, uncertainties=.99)
    # train perceptron
    sequences = readTest()
    trainPerceptron("test"+str(layersNumber)+"Layers", perceptron, loopNumber)
    pass
# define test
class TestBackPropagationDigits(unittest.TestCase):
    def test2Layers(self):
        testMultipleLayers(2, int(1e3))
        pass
    pass
    def test3Layers(self):
        testMultipleLayers(3, int(5e2))
        pass
    def test4Layers(self):
        testMultipleLayers(4, int(5e2))
        pass
    def test10Layers(self):
        testMultipleLayers(10, int(5e3))
        pass
    pass
    def test11Layers(self):
        testMultipleLayers(11, int(1e4))
        pass
    pass
    def test20Layers(self):
        testMultipleLayers(20, int(1e4))
        pass
    pass
pass
