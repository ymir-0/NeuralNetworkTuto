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
    def testDigits(self):
        # initialize perceptron
        layerHeights = ((30, 10))
        perceptron = Perceptron(layerHeights=layerHeights, weightLimit=1, uncertainties=.99)
        # train perceptron
        sequences=readTest(INPUT_DIRECTORY)
        trainPerceptron("testDigits",perceptron,sequences,int(1e3))
        pass
    pass
pass
