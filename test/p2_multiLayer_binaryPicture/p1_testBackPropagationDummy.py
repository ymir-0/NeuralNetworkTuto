#!/usr/bin/env python3
# PY test script file name must start with "test" to allow automatic recognition by PyCharm
# import
import unittest
from numpy import array
from neuralnetworktuto.p1_perceptron.p2_multiLayer_binaryPicture.p1_testBackPropagation import Perceptron, MetaParameters
from matplotlib.pyplot import plot, title , xlabel , ylabel, grid, legend, show
# define test
class TestHyperoperation(unittest.TestCase):
    # train perceptron
    @staticmethod
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
    # 1 hidden layer , 2 neurons on each layer, no random
    def testDummy0(self):
        # initialize perceptron
        weights = ((
            array(((
                ((0.15, 0.2)),
                ((0.25, 0.3)),
            ))),
            array(((
                ((0.4, 0.45)),
                ((0.5, 0.55)),
            ))),
        ))
        biases = ((0.35, 0.6))
        perceptron = Perceptron(weights=weights, biases=biases, uncertainties=.99)
        # train perceptron
        # INFO : numbers are assigned without any real logic
        sequences = dict({
            ((0.05, 0.1)): ((0.01, 0.99)),
            ((0.05, 0.01)): ((0.1, 0.99)),
            ((0.05, 0.99)): ((0.01, 0.1)),
            ((0.1, 0.05)): ((0.99, 0.01)),
            ((0.1, 0.99)): ((0.05, 0.01)),
            ((0.1, 0.01)): ((0.99, 0.05)),
            ((0.01, 0.99)): ((0.05, 0.1)),
            ((0.01, 0.05)): ((0.99, 0.1)),
            ((0.01, 0.1)): ((0.05, 0.99)),
            ((0.99, 0.01)): ((0.1, 0.05)),
        })
        TestHyperoperation.trainPerceptron("dummy0",perceptron,sequences,int(6e4))
    # 2 hidden layers , 2 neurons on each layer, no random
    def testDummy1(self):
        # initialize perceptron
        weights = ((
            array(((
                ((0.15, 0.2)),
                ((0.25, 0.3)),
            ))),
            array(((
                ((0.275, 0.325)),
                ((0.375, 0.425)),
            ))),
            array(((
                ((0.4, 0.45)),
                ((0.5, 0.55)),
            ))),
        ))
        biases = ((0.35, 0.475, 0.6))
        perceptron = Perceptron(weights=weights, biases=biases, uncertainties=.99)
        # train perceptron
        # INFO : numbers are assigned without any real logic
        sequences = dict({
            ((0.05, 0.1)): ((0.01, 0.99)),
            ((0.05, 0.01)): ((0.1, 0.99)),
            ((0.05, 0.99)): ((0.01, 0.1)),
            ((0.1, 0.05)): ((0.99, 0.01)),
            ((0.1, 0.99)): ((0.05, 0.01)),
            ((0.1, 0.01)): ((0.99, 0.05)),
            ((0.01, 0.99)): ((0.05, 0.1)),
            ((0.01, 0.05)): ((0.99, 0.1)),
            ((0.01, 0.1)): ((0.05, 0.99)),
            ((0.99, 0.01)): ((0.1, 0.05)),
            ((0.99, 0.1)): ((0.01, 0.05)),
            ((0.99, 0.05)): ((0.1, 0.01)),
        })
        TestHyperoperation.trainPerceptron("dummy1",perceptron, sequences, int(6.5e4))
    # 3 hidden layers , 2 neurons on each layer, randomized, all extra parameters updated on training
    def testDummy2(self):
        # WARNING : some randomized choice may not converge
        # initialize perceptron
        layerHeights = tuple([2] * 4)
        perceptron = Perceptron(layerHeights=layerHeights, uncertainties=.99)
        # train perceptron
        # INFO : numbers are assigned without any real logic
        sequences = dict({
            ((0.05, 0.1)): ((0.01, 0.99)),
            ((0.05, 0.01)): ((0.1, 0.99)),
            ((0.05, 0.99)): ((0.01, 0.1)),
            ((0.1, 0.05)): ((0.99, 0.01)),
            ((0.1, 0.99)): ((0.05, 0.01)),
            ((0.1, 0.01)): ((0.99, 0.05)),
            ((0.01, 0.99)): ((0.05, 0.1)),
            ((0.01, 0.05)): ((0.99, 0.1)),
            ((0.01, 0.1)): ((0.05, 0.99)),
            ((0.99, 0.01)): ((0.1, 0.05)),
            ((0.99, 0.1)): ((0.01, 0.05)),
            ((0.99, 0.05)): ((0.1, 0.01)),
        })
        TestHyperoperation.trainPerceptron("dummy2",perceptron, sequences, int(6e4))
        pass
    # 4 neurons on input, 3 neurons on single hidden one, 2 neurons on output one, random
    def testDummy3(self):
        # WARNING : some randomized choice may not converge
        # initialize perceptron
        layerHeights = ((4, 3, 2))
        perceptron = Perceptron(layerHeights=layerHeights, weightLimit=10, uncertainties=.99)
        # train perceptron
        # INFO : input is 4 numbers, output is mean & standard deviation
        sequences = dict({
            ((0.71, 0.21, 0.17, 0.26)): ((0.28, 0.04)),
            ((0.14, 0.11, 0.81, 0.01)): ((0.11, 0.32)),
            ((0.91, 0.83, 0.79, 0.33)): ((0.67, 0.2)),
            ((0.37, 0.86, 0.09, 0.9)): ((0.4, 0.34)),
            ((0.71, 0.77, 0.55, 0.32)): ((0.56, 0.16)),
            ((0.08, 0.8, 0.31, 0.88)): ((0.36, 0.25)),
            ((0.47, 0.08, 0.48, 0.42)): ((0.3, 0.15)),
            ((0.36, 0.1, 0.32, 0.23)): ((0.23, 0.08)),
            ((0.33, 0.94, 0.42, 0.81)): ((0.57, 0.2)),
            ((0.58, 0.7, 0.17, 0.86)): ((0.49, 0.26)),
        })
        TestHyperoperation.trainPerceptron("dummy3",perceptron,sequences,int(5e3))
        pass
    pass
pass
