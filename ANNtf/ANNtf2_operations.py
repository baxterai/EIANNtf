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
import math

#if(useBinaryWeights) or if(generateFirstLayerSDR)
generateLargeNetwork = True	#default: True
if(generateLargeNetwork):
	maximumNetworkHiddenLayerNeuronsAsFractionOfInputNeurons = 3.0	#default: 3.0	#2.0 
	generateNetworkNonlinearConvergence = True
else:
	#maximumNetworkHiddenLayerNeuronsAsFractionOfInputNeurons = 0.5
	#generateNetworkNonlinearConvergence = True
	maximumNetworkHiddenLayerNeuronsAsFractionOfInputNeurons = 0.8	#default: 0.8
	generateNetworkNonlinearConvergence = False	#default: False

	
if(generateNetworkNonlinearConvergence):
	networkDivergenceType = "nonLinearConverging"
	
	#if(applyNeuronThresholdBias):
	#	#this will affect the optimimum convergence angle
	#	networkOptimumConvergenceAngle = 0.5+applyNeuronThresholdBiasValue
		
	networkOptimumConvergenceAngle = 0.5	#if angle > 0.5, then more obtuse triange, if < 0.5 then more acute triangle	#fractional angle between 0 and 90 degrees
	networkDivergence = 1.0-networkOptimumConvergenceAngle 
	#required for Logarithms with a Fraction as Base:
	networkDivergenceNumerator = int(networkDivergence*10)
	networkDivergenceDenominator = 10
else:
	networkDivergenceType = "linearConverging"
	#networkDivergenceType = "linearDivergingThenConverging"	#not yet coded
	
	

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

def crossEntropy(y_pred, y_true, datasetNumClasses, costCrossEntropyWithLogits=False, oneHotEncoded=False):
	if(costCrossEntropyWithLogits):
		cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(y_pred), labels=tf.cast(y_true, tf.float32)))
		return cost
	else:
		if(not oneHotEncoded):
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

def filterNParraysByClassTargetInverse(train_x, train_y, classTargetFilterIndex=-1):
	rowFilter = (train_y != classTargetFilterIndex)
	#print("rowFilter = ", rowFilter)
	train_xFiltered = train_x[rowFilter]
	train_yFiltered = train_y[rowFilter]
	return train_xFiltered, train_yFiltered
	
def generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize):
	#shuffleSize = shuffleBufferSize
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


def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet):

	#Network parameters
	n_h = []
	numberOfLayers = 0
	numberOfNetworks = 0
	datasetNumClasses = 0
	
	numberOfNetworks = numberOfNetworksSet

	if((networkDivergenceType == "linearConverging") or (networkDivergenceType == "nonLinearConverging")):
		firstHiddenLayerNumberNeurons = int(num_input_neurons*maximumNetworkHiddenLayerNeuronsAsFractionOfInputNeurons)
	
	if(generateNetworkNonlinearConvergence):
		#(networkDivergenceType == "nonLinearConverging")
		#num_output_neurons = firstHiddenLayerNumberNeurons * networkDivergence^numLayers [eg 5 = 100*0.6^x]
		#if a^c = b, then c = log_a(b)
		b = float(num_output_neurons)/firstHiddenLayerNumberNeurons
		#numLayers = math.log(b, networkDivergence)
		
		#now log_a(x) = log_b(x)/log_b(a)
		#therefore log_a1/a2(b) = log_a2(b)/log_a2(a1/a2) = log_a2(b)/(log_a2(a1) - b)
		numberOfLayers = math.log(b, networkDivergenceDenominator)/math.log(float(networkDivergenceNumerator)/networkDivergenceDenominator, networkDivergenceDenominator)
		numberOfLayers = int(numberOfLayers)+1	#plus input layer
		
		print("numberOfLayers = ", numberOfLayers)
		
	else:
		if(dataset == "POStagSequence"):
			if(trainMultipleFiles):
				numberOfLayers = 6
			else:
				numberOfLayers = 3
		elif(dataset == "SmallDataset"):
			if(trainMultipleFiles):
				numberOfLayers = 6	#trainMultipleFiles should affect number of neurons/parameters in network
			else:
				numberOfLayers = 3

	n_x = num_input_neurons #datasetNumFeatures
	n_y = num_output_neurons  #datasetNumClasses
	datasetNumClasses = n_y
	n_h_first = n_x
	previousNumberLayerNeurons = n_h_first
	n_h.append(n_h_first)

	for l in range(1, numberOfLayers):	#for every hidden layer
		if(networkDivergenceType == "linearConverging"):
			if(l == 1):
				n_h_x = firstHiddenLayerNumberNeurons
			else:
				n_h_x = int((firstHiddenLayerNumberNeurons-num_output_neurons) * ((l-1)/(numberOfLayers-2)) + num_output_neurons)
			#previousNumberLayerNeurons = n_h_x
			n_h.append(n_h_x)
		elif(networkDivergenceType == "nonLinearConverging"):
			if(l == 1):
				n_h_x = firstHiddenLayerNumberNeurons
			else:
				n_h_x = int(previousNumberLayerNeurons*networkDivergence)
			n_h.append(n_h_x)
			previousNumberLayerNeurons = n_h_x
		elif(networkDivergenceType == "linearDivergingThenConverging"):
			#not yet coded
			print("defineNetworkParametersANN error: linearDivergingThenConverging not yet coded")
			exit()
		else:
			print("defineNetworkParametersANN error: unknown networkDivergenceType")
			exit()

	n_h_last = n_y
	n_h.append(n_h_last)
	
	print("defineNetworkParameters, n_h = ", n_h)
	
	return 	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses
	
