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



def generateParameterNameNetwork(networkIndex, l, arrayName):
	parameterName = "n" + str(networkIndex) + "l" + str(l) + arrayName
	return parameterName
	
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

def crossEntropy(y_pred, y_true, datasetNumClasses, costCrossEntropyWithLogits=False):
	if(costCrossEntropyWithLogits):
		cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(y_pred), labels=tf.cast(y_true, tf.float32)))
		return cost
	else:
		#print("datasetNumClasses = ", datasetNumClasses)
		#print("y_true = ", y_true)
		#print("y_pred = ", y_pred)
		y_true = tf.one_hot(y_true, depth=datasetNumClasses)
		y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
		cost = tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))
		return cost

def calculateAccuracy(y_pred, y_true):
	#print("y_pred.shape = ", y_pred.shape)
	#print("y_true.shape = ", y_true.shape)
	correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
	return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)
	
