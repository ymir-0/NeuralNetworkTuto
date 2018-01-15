#!/usr/bin/env python3
''' examples from :
 - https://stackoverflow.com/questions/2480650/role-of-bias-in-neural-networks
 - https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
'''
# imports
from numpy import exp, newaxis, zeros
from numpy.random import rand
from os import linesep, sep, listdir, makedirs
from os.path import realpath, join, exists
from random import shuffle
from shutil import rmtree
# contants
CURRENT_DIRECTORY = realpath(__file__).rsplit(sep, 1)[0]
OUTPUT_DIRECTORY = join(CURRENT_DIRECTORY,"output")
# tools classes
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
# perceptron
class Perceptron():
    def __init__(self,layerHeights,weights=None,biases=None,weightLimit=0.125,biasLimit=1):
        # TODO : check dimensions consistancy if not random
        # TODO : set global weight/biais/uncertainty/dilatation/offset for all neurons or make it specific for each layer
        Logger.append(0, "initializing perceptron from layer heights : " + str(layerHeights))
        # initialize attributs
        hiddenLayerNumbers=len(layerHeights)-1
        # initialize weights...
        # ... from input
        if weights is not None:
            Logger.append(0, "from given weights : " + str(weights))
            self.weights = weights
            pass
        # ... randomly
        else:
            Logger.append(0, "random weights in range : " + str(weightLimit))
            # for each layer
            # INFO : there is no weights related to input layer
            self.weights = list()
            for layerIndex in range(0, hiddenLayerNumbers):
                # randomize weights
                self.randomizeLayer(layerIndex, weightLimit)
                pass
            pass
        # initialize weights...
        # ... from input
        if biases is not None:
            Logger.append(0, "from given biases : " + str(biases))
            self.biases = biases
        # ... randomly
        else:
            Logger.append(0, "random biases in range : " + str(biasLimit))
            self.biases = (rand(hiddenLayerNumbers)-.5)*2*biasLimit
        #
        Logger.append(0, "initialized perceptron" + linesep + str(self))
        pass
    def randomizeLayer(self, layerIndex,weightLimit):
        # get heights for current & previous layers
        currentHeight = layerHeights[layerIndex]
        previousHeight = layerHeights[layerIndex-1]
        # randomize layer weights
        layerWeights=(rand(currentHeight,previousHeight)-.5)*2*weightLimit
        Logger.append(1, "initialized layer weights #" + str(layerIndex) + linesep + str(layerWeights))
        self.weights.append(layerWeights)
        pass
    pass
    def __str__(self):
        return "weights : " + linesep + str(self.weights) + linesep + "biases : " + linesep + str(self.biases)
        pass
    pass
pass
# empty output folder
if exists(OUTPUT_DIRECTORY):
    rmtree(OUTPUT_DIRECTORY)
makedirs(OUTPUT_DIRECTORY)
# perceptron initialization
layerHeights=((2,2,2))
weights=list()
weights.append(zeros((2,2)))
weights.append(zeros((2,2)))
weights[0][0][0]=0.15
weights[0][0][1]=0.25
weights[0][1][0]=0.2
weights[0][1][1]=0.3
weights[1][0][0]=0.4
weights[1][0][1]=0.5
weights[1][1][0]=0.45
weights[1][1][1]=0.55
biases=((0.35,0.6))
perceptron = Perceptron(layerHeights,weights,biases)
# flush logs
Logger.flush()
pass