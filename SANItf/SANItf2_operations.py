# -*- coding: utf-8 -*-
"""SANItf2_operations.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Usage:
see SANItf2.py

# Description

SANI operations

- Author: Richard Bruce Baxter - Copyright (c) 2020 Baxter AI (baxterai.com)

"""

import tensorflow as tf
import numpy as np
import SANItf2_globalDefs


def generateParameterNameSeq(l, s, arrayName):
	parameterName = "l" + str(l) + arrayName + "s" + str(s)
	return parameterName
	
def generateParameterName(l, arrayName):
	parameterName = "l" + str(l) + arrayName
	return parameterName
	
def printShape(tensor, tensorName):
	print(tensorName + ".shape = ")
	print(tensor.shape)
	
def printAverage(tensor, tensorName, indentation):
	tensorAverage = tf.reduce_mean(tf.dtypes.cast(tensor, tf.float32))
	indentationString = ""
	for i in range(indentation):
		indentationString = indentationString + "\t"
	print(indentationString + tensorName + "Average: %f" % (tensorAverage))
