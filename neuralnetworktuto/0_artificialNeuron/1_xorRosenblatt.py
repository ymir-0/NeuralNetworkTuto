#!/usr/bin/env python3
# see : https://www.anyflo.com/bret/cours/rn/rn3.htm
# imports
from numpy import heaviside, array, append
# neuron
class Neuron():
    def __init__(self,weights,threshold):
        self.thresholdedWeights=append(weights,-threshold)
        pass
    def activate(self,inputs):
        # sum weighted input
        thresholdedInputs=append(inputs,1)
        weightedInputs = self.thresholdedWeights.dot(thresholdedInputs.transpose())
        # compute & return OUT
        output = heaviside(weightedInputs, 1)
        return output
andNeuron=Neuron(array([.75,.75]),1) # w&A=w&B=.75 ; T&=1
xorNeuron=Neuron(array([1.25,1.25,-1.75]),1) # wXA=wXB=1.25; wX&=-1.75 ; TX=1
# network
def xorNetwork(a,b):
    # and
    andInput=array([a,b])
    andOutput=andNeuron.activate(andInput)
    # xor & return
    xorInput=append(andInput,andOutput)
    xorOutput=xorNeuron.activate(xorInput)
    return xorOutput
pass
# test
print("xor 0 0 : " + str(xorNetwork(0,0)))
print("xor 0 1 : " + str(xorNetwork(0,1)))
print("xor 1 0 : " + str(xorNetwork(1,0)))
print("xor 1 1 : " + str(xorNetwork(1,1)))
