# -*- coding: utf-8 -*-
"""ANNtf2_operations.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Usage:
see ANNtf2.py

# Description

SANI operations

- Author: Richard Bruce Baxter - Copyright (c) 2020 Baxter AI (baxterai.com)

"""

import tensorflow as tf
import numpy as np
import ANNtf2_globalDefs



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
		y_true = tf.one_hot(y_true, depth=datasetNumClasses)
		y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
		cost = tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))
		return cost

def calculateAccuracy(y_pred, y_true):
	correct_prediction = calculateCorrectPrediction(y_pred, y_true) 
	return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)
	
def calculateCorrectPrediction(y_pred, y_true):
	correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
	return correct_prediction

def filterNParraysByClassTarget(train_x, train_y, classTargetFilterIndex=-1):
	rowFilter = (train_y == classTargetFilterIndex)
	#print("rowFilter = ", rowFilter)
	train_xFiltered = train_x[rowFilter]
	train_yFiltered = train_y[rowFilter]
	return train_xFiltered, train_yFiltered

def generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize):
	trainDataUnbatched = generateTFtrainDataUnbatchedFromNParrays(train_x, train_y)
	trainData = generateTFtrainDataFromTrainDataUnbatched(trainDataUnbatched, shuffleSize, batchSize)
	return trainData

def generateTFtrainDataUnbatchedFromNParrays(train_x, train_y):
	#print("train_x.shape = ", train_x.shape)
	#print("train_y.shape = ", train_y.shape)
	trainDataUnbatched = tf.data.Dataset.from_tensor_slices((train_x, train_y))
	return trainDataUnbatched

def generateTFtrainDataFromTrainDataUnbatched(trainDataUnbatched, shuffleSize, batchSize):
	trainData = trainDataUnbatched.repeat().shuffle(shuffleSize).batch(batchSize).prefetch(1)	#do not repeat
	return trainData

