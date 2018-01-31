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
def readTest(dataFolder):
    # initialize data
    sequences = dict()
    # for each data file
    for dataFileShortName in listdir(dataFolder):
        # extract data key
        expectedDigit = int(dataFileShortName.split(".")[0])
        digits = [0]*10
        digits[expectedDigit] = 1
        digits = tuple(digits)
        # read it
        dataFileFullName = join(dataFolder, dataFileShortName)
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
def trainPerceptron(name, perceptron, sequences, loopNumber, metaParametersUpdate=((MetaParameters.BIASES.value))):
    # train perceptron
    print("test : " + name)
    print("loop number = " + str(loopNumber))
    errors = perceptron.train(sequences, loopNumber, metaParametersUpdate=metaParametersUpdate)
    # display results
    for input, expectedOutput in sequences.items():
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
# define test
class TestBackPropagationDigits(unittest.TestCase):
    def test0Hidden(self):
        # initialize perceptron
        layerHeights = ((30, 10))
        perceptron = Perceptron(layerHeights=layerHeights, weightLimit=1, uncertainties=.99)
        # train perceptron
        sequences=readTest(INPUT_DIRECTORY)
        trainPerceptron("testDigits",perceptron,sequences,int(1e3))
        pass
    pass
    def test1Hidden(self):
        # initialize perceptron
        layerHeights = ((30, 20, 10))
        perceptron = Perceptron(layerHeights=layerHeights, weightLimit=1, uncertainties=.99)
        # train perceptron
        sequences = readTest(INPUT_DIRECTORY)
        trainPerceptron("test1Hidden", perceptron, sequences, int(5e2))
        pass
    def test2Hidden(self):
        # initialize perceptron
        layerHeights = ((30, 23, 16, 10))
        perceptron = Perceptron(layerHeights=layerHeights, weightLimit=1, uncertainties=.99)
        # train perceptron
        sequences = readTest(INPUT_DIRECTORY)
        trainPerceptron("test2Hidden", perceptron, sequences, int(5e2))
        pass
    def test10Hidden(self):
        # initialize perceptron
        layerHeights = list()
        for layerHeight in range(0,10):
            layerHeights.append(int(layerHeight*20/9+10))
        layerHeights.reverse()
        layerHeights = tuple(layerHeights)
        perceptron = Perceptron(layerHeights=layerHeights, weightLimit=1, uncertainties=.99)
        # train perceptron
        sequences = readTest(INPUT_DIRECTORY)
        trainPerceptron("test10Hidden", perceptron, sequences, int(5e3))
        pass
    pass
    def test11Hidden(self):
        # initialize perceptron
        layerHeights = list()
        for layerHeight in range(0,11):
            layerHeights.append(int(layerHeight*20/10+10))
        layerHeights.reverse()
        layerHeights = tuple(layerHeights)
        perceptron = Perceptron(layerHeights=layerHeights, weightLimit=1, uncertainties=.99)
        # train perceptron
        sequences = readTest(INPUT_DIRECTORY)
        trainPerceptron("test11Hidden", perceptron, sequences, int(1e4))
        pass
    pass
    def test13Hidden(self):
        # initialize perceptron
        layerHeights = list()
        for layerHeight in range(0,13):
            layerHeights.append(int(layerHeight*20/12+10))
        layerHeights.reverse()
        layerHeights = tuple(layerHeights)
        perceptron = Perceptron(layerHeights=layerHeights, weightLimit=1, uncertainties=.99)
        # train perceptron
        sequences = readTest(INPUT_DIRECTORY)
        trainPerceptron("test13Hidden", perceptron, sequences, int(1e4))
        pass
    pass
    def test15Hidden(self):
        # initialize perceptron
        layerHeights = list()
        for layerHeight in range(0,15):
            layerHeights.append(int(layerHeight*20/14+10))
        layerHeights.reverse()
        layerHeights = tuple(layerHeights)
        perceptron = Perceptron(layerHeights=layerHeights, weightLimit=1, uncertainties=.99)
        # train perceptron
        sequences = readTest(INPUT_DIRECTORY)
        trainPerceptron("test15Hidden", perceptron, sequences, int(1e4))
        pass
    pass
    def test20Hidden(self):
        # initialize perceptron
        layerHeights = list()
        for layerHeight in range(10,31):
            layerHeights.append(layerHeight)
        layerHeights.reverse()
        layerHeights = tuple(layerHeights)
        perceptron = Perceptron(layerHeights=layerHeights, weightLimit=1, uncertainties=.99)
        # train perceptron
        sequences = readTest(INPUT_DIRECTORY)
        trainPerceptron("test20Hidden", perceptron, sequences, int(1e4))
        pass
    pass
pass
