# -*- coding: utf-8 -*-
"""SANItf2.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Usage:
python3 SANItf2.py

# Description

Train an artificial neural network (ANN or SANI)

- Author: Richard Bruce Baxter - Copyright (c) 2020 Baxter AI (baxterai.com)

"""

# %tensorflow_version 2.x
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import tensorflow as tf
import numpy as np

import sys
np.set_printoptions(threshold=sys.maxsize)

import SANItf2_algorithmSANI
import SANItf2_algorithmANN
from SANItf2_loadDataset import loadDatasetType1, loadDatasetType2

algorithm = "SANI"
#algorithm = "ANN"
if(algorithm == "SANI"):
	dataset = "POStagSequence"
elif(algorithm == "ANN"):
	dataset = "POStagSequence"
	#dataset = "NewThyroid"

trainMultipleFiles = False
datasetFileNameXstart = "XtrainBatch"
datasetFileNameYstart = "YtrainBatch"
datasetFileNameXend = ".dat"
datasetFileNameYend = ".dat"
datasetFileNameStart = "trainBatch"
datasetFileNameEnd = ".dat"


#Training parameters
learningRate = 0.001
if(dataset == "POStagSequence"):
	trainingSteps = 10000
elif(dataset == "NewThyroid"):
	trainingSteps = 1000

if(SANItf2_algorithmSANI.tMinMidMaxUpdateMode == "fastApproximation"):
	batchSize = 1000
	displayStep = 100
elif(SANItf2_algorithmSANI.tMinMidMaxUpdateMode == "slowExact"):
	batchSize = 50
	displayStep = 10


def neuralNetworkPropagation(x):
	if(algorithm == "SANI"):
		pred = SANItf2_algorithmSANI.neuralNetworkPropagationSANI(x)
	elif(algorithm == "ANN"):
		pred = SANItf2_algorithmANN.neuralNetworkPropagationANN(x)
	return pred
	
def crossEntropy(y_pred, y_true):
	y_true = tf.one_hot(y_true, depth=datasetNumClasses)
	y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
	return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))

def calculateAccuracy(y_pred, y_true):
	correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
	return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

def executeOptimisation(x, y):
	with tf.GradientTape() as g:
		pred = neuralNetworkPropagation(x)
		loss = crossEntropy(pred, y)
		
	if(algorithm == "SANI"):
		trainableVariables = list(SANItf2_algorithmSANI.Wseq.values())  + list(SANItf2_algorithmSANI.Bseq.values())
		#trainableVariables = list(W.values()) + list(B.values()) + list(Wseq.values())  + list(Bseq.values())
	elif(algorithm == "ANN"):
		trainableVariables = list(SANItf2_algorithmANN.W.values())  + list(SANItf2_algorithmANN.B.values())

	gradients = g.gradient(loss, trainableVariables)
	
	optimizer.apply_gradients(zip(gradients, trainableVariables))




#generate network parameters based on dataset properties:

datasetNumExamplesTemp = 0
datasetNumFeatures = 0
datasetNumClasses = 0

fileIndexTemp = 0
fileIndexStr = str(fileIndexTemp).zfill(4)
datasetType1FileNameX = datasetFileNameXstart + fileIndexStr + datasetFileNameXend
datasetType1FileNameY = datasetFileNameYstart + fileIndexStr + datasetFileNameYend
datasetType2FileName = datasetFileNameStart + fileIndexStr + datasetFileNameEnd

if(dataset == "POStagSequence"):
	datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = loadDatasetType1(datasetType1FileNameX, datasetType1FileNameY)
elif(dataset == "NewThyroid"):
	datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = loadDatasetType2(datasetType2FileName)

#Model constants
num_input_neurons = datasetNumFeatures  #train_x.shape[1]
num_output_neurons = datasetNumClasses

#print("datasetNumFeatures = " + str(datasetNumFeatures))
#print(train_xTemp[0:5,:])
#print(train_yTemp[0:5])
#print(test_xTemp[0:5,:])
#print(test_yTemp[0:5])

if(algorithm == "SANI"):
	SANItf2_algorithmSANI.defineNetworkParametersSANI(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset)
	SANItf2_algorithmSANI.defineNeuralNetworkParametersSANI()
elif(algorithm == "ANN"):
	SANItf2_algorithmANN.defineNetworkParametersANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset)
	SANItf2_algorithmANN.defineNeuralNetworkParametersANN()
		
		
#define epochs:
			
numEpochs = -1
fileIndexFirst = -1
fileIndexLast = -1
if(trainMultipleFiles):
	numEpochs = 10
	fileIndexFirst = 0
	fileIndexLast = 99
else:
	numEpochs = 1
	fileIndexFirst = 0 
	fileIndexLast = 0


# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learningRate)

for e in range(numEpochs):

	fileIndexArray = np.arange(fileIndexFirst, fileIndexLast+1, 1)
	#print("fileIndexArray = " + str(fileIndexArray))
	np.random.shuffle(fileIndexArray)
	fileIndexShuffledArray = fileIndexArray
	#print("fileIndexShuffledArray = " + str(fileIndexShuffledArray))
	
	for fileIndex in fileIndexShuffledArray:	#range(fileIndexFirst, fileIndexLast+1):
				
		datasetNumExamples = 0

		fileIndexStr = str(fileIndex).zfill(4)
		datasetType1FileNameX = datasetFileNameXstart + fileIndexStr + datasetFileNameXend
		datasetType1FileNameY = datasetFileNameYstart + fileIndexStr + datasetFileNameYend
		datasetType2FileName = datasetFileNameStart + fileIndexStr + datasetFileNameEnd
		
		if(dataset == "POStagSequence"):
			datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = loadDatasetType1(datasetType1FileNameX, datasetType1FileNameY)
		elif(dataset == "NewThyroid"):
			datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = loadDatasetType2(datasetType2FileName)

		shuffleSize = datasetNumExamples	#10*batchSize
		trainData = tf.data.Dataset.from_tensor_slices((train_x, train_y))
		trainData = trainData.repeat().shuffle(shuffleSize).batch(batchSize).prefetch(1)	#do not repeat

		for batchIndex, (batchX, batchY) in enumerate(trainData.take(trainingSteps), 1):
			
			executeOptimisation(batchX, batchY)

			if batchIndex % displayStep == 0:
				pred = neuralNetworkPropagation(batchX)
				loss = crossEntropy(pred, batchY)
				acc = calculateAccuracy(pred, batchY)
				print("batchIndex: %i, loss: %f, accuracy: %f" % (batchIndex, loss, acc))

		pred = neuralNetworkPropagation(test_x)
		print("Test Accuracy: %f" % calculateAccuracy(pred, test_y))
