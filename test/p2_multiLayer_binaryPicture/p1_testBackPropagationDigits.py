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
from networkx import Graph, get_node_attributes, draw, draw_networkx_labels
from matplotlib import cm
from matplotlib.pyplot import text, xlim, ylim, close
# contants
CURRENT_DIRECTORY = realpath(__file__).rsplit(sep, 1)[0]
INPUT_DIRECTORY = join(CURRENT_DIRECTORY,"input")
OUTPUT_DIRECTORY = join(CURRENT_DIRECTORY,"output")
# utility methods
def readInputs(label,readFunction):
    # initialize data
    sequences = dict()
    # for each data file
    digitDirectory = join(INPUT_DIRECTORY , label)
    for dataFileShortName in listdir(digitDirectory):
        partName=readFunction(dataFileShortName)
        # read it
        dataFileFullName = join(digitDirectory, dataFileShortName)
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
        sequences[image] = partName
    # return
    return sequences
def readDigits(dataFileName):
    # get digit name
    expectedDigit = int(dataFileName.split(".")[0])
    digits = [0]*10
    digits[expectedDigit] = 1
    digits = tuple(digits)
    # return
    return digits
DIGITS = readInputs("digit",readDigits)
def readParts(dataFileName):
    # get part name
    partName=dataFileName.split(".")[0]
    # return
    return partName
def generateLayerHeights(layersNumber):
    layerHeights =[int(layerHeight * 20 / (layersNumber-1) + 10) for layerHeight in range(0, layersNumber)]
    layerHeights.reverse()
    return tuple(layerHeights)
def trainPerceptron(layersNumber, perceptron, loopNumber, metaParametersUpdate=((MetaParameters.BIASES.value))):
    # initialize report
    report = ""
    # train perceptron
    report = report + "test : " + str(layersNumber) + " layers" + linesep
    report = report + "loop number : " + str(loopNumber) + linesep
    errors = perceptron.train(DIGITS, loopNumber, metaParametersUpdate=metaParametersUpdate)
    # display results
    figure()
    for input, expectedOutput in DIGITS.items():
        # print results
        actualOutput = perceptron.passForward(input)
        report = report + "input = " + str(input) + "\texpected output = " + str(expectedOutput) + "\tactual output = " + str(
            actualOutput) + "\terror = " + str(errors[expectedOutput][-1]) + linesep
        # get related digit
        digit = expectedOutput.index(1)
        # prepare graph
        plot(errors["loopNumbers"], errors[expectedOutput], label=str(digit))
    # set output file
    reportFolder = join(OUTPUT_DIRECTORY, str(layersNumber) + "layers", "training")
    makedirs(reportFolder)
    reportFile = join(reportFolder,"errorEvolution")
    # write report
    writeTrainingReport(reportFile,report)
    # plot graph
    title("errors evolution ( " + str(layersNumber) + " layers)")
    xlabel("training step")
    ylabel("error")
    grid(linestyle="-.")
    legend()
    saveFigure(reportFile)
    pass
def writeTrainingReport(name,report):
    filePath = name + ".txt"
    file = open(filePath, "wt")
    file.write(report)
    file.close()
def saveFigure(name):
    figurePath = name + ".png"
    savefig(figurePath)
    close()
def testMapNeuronsActivation(layersNumber, perceptron):
    # set report folder
    reportFolder = join(OUTPUT_DIRECTORY, str(layersNumber) + "layers", "neuronsActivations", "digits")
    makedirs(reportFolder)
    # activate all digits
    for inputOutput, expectedOutput in DIGITS.items():
        # get related digit
        digit = expectedOutput.index(1)
        # wite report
        writeNeuronsActivationReport(layersNumber, perceptron, inputOutput, digit, reportFolder)
        pass
    pass
def writeNeuronsActivationReport(layersNumber,perceptron,inputOutput,expectedOutput, reportFolder):
    # initialize graph
    graph = Graph()
    addNodesToGraph(graph, 0, inputOutput)
    # for each layer
    for layerIndex, layer in enumerate(perceptron.layers):
        inputOutput = layer.passForward(inputOutput)
        addNodesToGraph(graph, layerIndex + 1, inputOutput)
    # draw graph
    figure(figsize=(10, 10))
    # INFO : positions & labels must be DICT type
    positions = get_node_attributes(graph, 'position')
    labels = get_node_attributes(graph, 'label')
    # INFO : intensities must be LIST type
    intensities = tuple([graph.nodes(data='intensity')[node] for node in graph.nodes])
    draw(graph, pos=positions, cmap=cm.Reds, node_color=intensities)
    draw_networkx_labels(graph, positions, labels)
    text(0, 1, "neurons activation for digit " + str(expectedOutput), size=15)
    ylim(-31, 2)
    # set output file
    reportFile = join(reportFolder,str(expectedOutput))
    saveFigure(reportFile)
pass
def addNodesToGraph(graph,layerIndex, layerValues):
    # for each layer value
    for nodeIndex, layerValue in enumerate(layerValues):
        # set node property
        # INFO : 'y' axis is reversed to read network from top to bottom
        position = (layerIndex, -nodeIndex)
        approximativeLayerValue=round(layerValue,2)
        label = "L"+str(layerIndex)+"N"+str(nodeIndex)+"V"+str(approximativeLayerValue)
        # add node
        # INFO : position is also key
        graph.add_node(position, position=position,label=label,intensity=approximativeLayerValue)
        pass
    pass
def testSearchConvolution(layersNumber,perceptron):
    # set report folder
    reportFolder = join(OUTPUT_DIRECTORY, str(layersNumber) + "layers", "neuronsActivations", "parts")
    makedirs(reportFolder)
    # run each part
    parts = readInputs("part", readParts)
    for inputOutput, expectedOutput in parts.items():
        # wite report
        writeNeuronsActivationReport(layersNumber, perceptron, inputOutput, expectedOutput, reportFolder)
        pass
    pass
def testErrorsEvolution():
    # INFO : from p1_binary_no_deterministic.py l.117 drawErrorsEvolution
    pass
pass
def testMultipleLayers(layersNumber,loopNumber):
    # create report folder
    reportFolder = join(OUTPUT_DIRECTORY, str(layersNumber) + "layers")
    makedirs(reportFolder)
    # initialize perceptron
    layerHeights = generateLayerHeights(layersNumber)
    perceptron = Perceptron(layerHeights=layerHeights, weightLimit=1, uncertainties=.99)
    #name = "test"+str(layersNumber)+"Layers"
    # train perceptron
    trainPerceptron(layersNumber, perceptron, loopNumber)
    # map neurons activation
    testMapNeuronsActivation(layersNumber, perceptron)
    # search convolutions
    testSearchConvolution(layersNumber, perceptron)
    pass
# empty output folder
if exists(OUTPUT_DIRECTORY):
    rmtree(OUTPUT_DIRECTORY)
makedirs(OUTPUT_DIRECTORY)
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
# TODO : blur each digit bit by bit and mesure errors evolution
# TODO : test loop for all layers & replace test class by a main method