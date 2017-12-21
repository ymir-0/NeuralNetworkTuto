#!/usr/bin/env python3
# see : https://www.anyflo.com/bret/cours/rn/rn3.htm
# imports
from numpy import heaviside, array
# constants
WA=-.75 # weight for input 'a'
WB=-.75 # weight for input 'b'
T=-1 # threshold
THRESHOLD=array((WA,WB,T))
# neuron
def neuron(a,b):
    # sum weighted input
    INPUT_ARRAY=array((a,b,-1))
    WEIGHTED_INPUT=THRESHOLD.dot(INPUT_ARRAY.transpose())
    # compute & return OUT
    OUTPUT=heaviside(WEIGHTED_INPUT, 1)
    return OUTPUT
# logical functions
nand = lambda a, b : neuron(a,b)
not_ = lambda a : nand(a,a)
and_ = lambda a, b : nand(nand(a,b),nand(a,b))
or_ = lambda a, b : nand(not_(a),not_(b))
nor_ = lambda a, b : not_(or_(a,b))
xor = lambda a, b : and_(or_(a,b),not_(and_(a,b)))
# ... we can build all logical functions
# test
print("nand 0 0 : " + str(nand(0,0)))
print("nand 0 1 : " + str(nand(0,1)))
print("nand 1 0 : " + str(nand(1,0)))
print("nand 1 1 : " + str(nand(1,1)))
print("not 0 : " + str(not_(0)))
print("not 1 : " + str(not_(1)))
print("and 0 0 : " + str(and_(0,0)))
print("and 0 1 : " + str(and_(0,1)))
print("and 1 0 : " + str(and_(1,0)))
print("and 1 1 : " + str(and_(1,1)))
print("or 0 0 : " + str(or_(0,0)))
print("or 0 1 : " + str(or_(0,1)))
print("or 1 0 : " + str(or_(1,0)))
print("or 1 1 : " + str(or_(1,1)))
print("nor 0 0 : " + str(nor_(0,0)))
print("nor 0 1 : " + str(nor_(0,1)))
print("nor 1 0 : " + str(nor_(1,0)))
print("nor 1 1 : " + str(nor_(1,1)))
print("xor 0 0 : " + str(xor(0,0)))
print("xor 0 1 : " + str(xor(0,1)))
print("xor 1 0 : " + str(xor(1,0)))
print("xor 1 1 : " + str(xor(1,1)))
pass
