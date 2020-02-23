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
#np.set_printoptions(threshold=sys.maxsize)

import SANItf2_globalDefs

algorithm = "SANI"
#algorithm = "ANN"
if(algorithm == "SANI"):
	algorithmSANI = "sharedModulesBinary"
	#algorithmSANI = "sharedModules"
	#algorithmSANI = "repeatedModules"
	if(algorithmSANI == "repeatedModules"):
		import SANItf2_algorithmSANIrepeatedModules as SANItf2_algorithmSANI
	elif(algorithmSANI == "sharedModules"):
		import SANItf2_algorithmSANIsharedModules as SANItf2_algorithmSANI
	elif(algorithmSANI == "sharedModulesBinary"):
		import SANItf2_algorithmSANIsharedModulesBinary as SANItf2_algorithmSANI
elif(algorithm == "ANN"):
	import SANItf2_algorithmANN

import SANItf2_loadDataset

#performance enhancements for development environment only: 
debugUseSmallDataset = True	#def:False	#switch increases performance during development	#eg data-POStagSentence-smallBackup
useSmallSentenceLengths = True	#def:False	#switch increases performance during development	#eg data-simple-POStagSentence-smallBackup
trainMultipleFiles = False	#def:True	#switch increases performance during development	#eg data-POStagSentence

#loadDatasetType3 parameters:
#if generatePOSunambiguousInput=True, generate POS unambiguous permutations for every POS ambiguous data example/experience
#if onlyAddPOSunambiguousInputToTrain=True, do not train network with ambiguous POS possibilities
#if generatePOSunambiguousInput=False and onlyAddPOSunambiguousInputToTrain=False, requires simultaneous propagation of different (ambiguous) POS possibilities

if(algorithm == "SANI"):
	if(algorithmSANI == "repeatedModules"):
		dataset = "POStagSequence"
	elif(algorithmSANI == "sharedModules"):
		dataset = "POStagSentence"
		numberOfFeaturesPerWord = -1
		paddingTagIndex = -1
		if(SANItf2_algorithmSANI.allowMultipleContributingSubinputsPerSequentialInput):
			generatePOSunambiguousInput = False
			onlyAddPOSunambiguousInputToTrain = False
		else:
			generatePOSunambiguousInput = False
			onlyAddPOSunambiguousInputToTrain = True
	elif(algorithmSANI == "sharedModulesBinary"):
		dataset = "POStagSentence"
		numberOfFeaturesPerWord = -1
		paddingTagIndex = -1	
		generatePOSunambiguousInput = False
		onlyAddPOSunambiguousInputToTrain = False	#True
elif(algorithm == "ANN"):
	#dataset = "POStagSequence"
	dataset = "NewThyroid"

if(debugUseSmallDataset):
	datasetFileNameXstart = "XtrainBatchSmall"
	datasetFileNameYstart = "YtrainBatchSmall"
else:
	datasetFileNameXstart = "XtrainBatch"
	datasetFileNameYstart = "YtrainBatch"
datasetFileNameXend = ".dat"
datasetFileNameYend = ".dat"
datasetFileNameStart = "trainBatch"
datasetFileNameEnd = ".dat"


	
		
def neuralNetworkPropagation(x):
	if(algorithm == "SANI"):
		pred = SANItf2_algorithmSANI.neuralNetworkPropagationSANI(x)
	elif(algorithm == "ANN"):
		pred = SANItf2_algorithmANN.neuralNetworkPropagationANN(x)
	return pred
	
def crossEntropy(y_pred, y_true):
	if((algorithm == "SANI") and (algorithmSANI == "sharedModules")):
		cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(y_pred), labels=tf.cast(y_true, tf.float32)))
		return cost
	else:
		y_true = tf.one_hot(y_true, depth=datasetNumClasses)
		y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
		cost = tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))
		return cost

def calculateAccuracy(y_pred, y_true):
	correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
	return tf.reduce_mean(tf.cast(correct_prediction, tf.float32), axis=-1)

def executeOptimisation(x, y):
	with tf.GradientTape() as g:
		pred = neuralNetworkPropagation(x)
		loss = crossEntropy(pred, y)
		
	if(algorithm == "SANI"):
		if(algorithmSANI == "sharedModules"):
			if(SANItf2_algorithmSANI.useSparseTensors):
				if(SANItf2_algorithmSANI.performSummationOfSubInputsWeighted):
					if(SANItf2_algorithmSANI.performSummationOfSequentialInputsWeighted):
						trainableVariables = list(SANItf2_algorithmSANI.W.values()) + list(SANItf2_algorithmSANI.Wseq.values())	#+ list(SANItf2_algorithmSANI.B.values()) 
					else:
						trainableVariables = list(SANItf2_algorithmSANI.Wseq.values())
				else:
					if(SANItf2_algorithmSANI.performSummationOfSequentialInputsWeighted):
						trainableVariables = list(SANItf2_algorithmSANI.W.values()) #+ list(SANItf2_algorithmSANI.B.values())
					else:
						trainableVariables = list()	
			else:
				if(SANItf2_algorithmSANI.performSummationOfSequentialInputsWeighted):
					trainableVariables = list(SANItf2_algorithmSANI.W.values()) + list(SANItf2_algorithmSANI.Wseq.values()) + list(SANItf2_algorithmSANI.Bseq.values())	#+ list(SANItf2_algorithmSANI.B.values()) 
				else:
					trainableVariables = list(SANItf2_algorithmSANI.Wseq.values()) + list(SANItf2_algorithmSANI.Bseq.values())
					#trainableVariables = list(SANItf2_algorithmSANI.Wseq.values())
					
		elif(algorithmSANI == "repeatedModules"):
			if(SANItf2_algorithmSANI.allowMultipleSubinputsPerSequentialInput):
				if(SANItf2_algorithmSANI.performSummationOfSequentialInputsWeighted):
					if(SANItf2_algorithmSANI.performSummationOfSubInputsWeighted):
						trainableVariables = list(SANItf2_algorithmSANI.W.values()) + list(SANItf2_algorithmSANI.Wseq.values())
						#trainableVariables = list(SANItf2_algorithmSANI.W.values()) + list(SANItf2_algorithmSANI.B.values()) + list(SANItf2_algorithmSANI.Wseq.values()) + list(SANItf2_algorithmSANI.Bseq.values())
					else:
						trainableVariables = list(SANItf2_algorithmSANI.W.values())
						#trainableVariables = list(SANItf2_algorithmSANI.W.values()) + list(SANItf2_algorithmSANI.B.values())
				else:
					if(SANItf2_algorithmSANI.performSummationOfSubInputsWeighted):
						trainableVariables = list(SANItf2_algorithmSANI.Wseq.values())
						#trainableVariables = list(SANItf2_algorithmSANI.Wseq.values()) + list(SANItf2_algorithmSANI.Bseq.values())
					else:
						print("error: allowMultipleSubinputsPerSequentialInput && !performSummationOfSequentialInputsWeighted && !performSummationOfSubInputsWeighted")
			else:
				if(SANItf2_algorithmSANI.performSummationOfSequentialInputsWeighted):
					trainableVariables = list(SANItf2_algorithmSANI.W.values())
					#trainableVariables = list(SANItf2_algorithmSANI.W.values()) + list(SANItf2_algorithmSANI.B.values())
				else:
					print("error: !allowMultipleSubinputsPerSequentialInput && !performSummationOfSequentialInputsWeighted")
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
	datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = SANItf2_loadDataset.loadDatasetType1(datasetType1FileNameX, datasetType1FileNameY)
elif(dataset == "POStagSentence"):
	numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = SANItf2_loadDataset.loadDatasetType3(datasetType1FileNameX, generatePOSunambiguousInput, onlyAddPOSunambiguousInputToTrain, useSmallSentenceLengths)
elif(dataset == "NewThyroid"):
	datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = SANItf2_loadDataset.loadDatasetType2(datasetType2FileName)



#Model constants
num_input_neurons = datasetNumFeatures  #train_x.shape[1]
num_output_neurons = datasetNumClasses

if(algorithm == "SANI"):
	if(dataset == "POStagSentence"):
		SANItf2_algorithmSANI.defineTrainingParametersSANIsharedModules(numberOfFeaturesPerWord, paddingTagIndex)
	learningRate, trainingSteps, batchSize, displayStep = SANItf2_algorithmSANI.defineTrainingParametersSANI(dataset, trainMultipleFiles)
	SANItf2_algorithmSANI.defineNetworkParametersSANI(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, useSmallSentenceLengths)
	SANItf2_algorithmSANI.defineNeuralNetworkParametersSANI()
elif(algorithm == "ANN"):
	learningRate, trainingSteps, batchSize, displayStep = SANItf2_algorithmANN.defineTrainingParametersANN(dataset, trainMultipleFiles)
	SANItf2_algorithmANN.defineNetworkParametersANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles)
	SANItf2_algorithmANN.defineNeuralNetworkParametersANN()
		
		
#define epochs:
			
numEpochs = -1
fileIndexFirst = -1
fileIndexLast = -1
if(trainMultipleFiles):
	numEpochs = 10
	fileIndexFirst = 0
	if(useSmallSentenceLengths):
		fileIndexLast = 11
	else:
		fileIndexLast = 1202
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
			datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = SANItf2_loadDataset.loadDatasetType1(datasetType1FileNameX, datasetType1FileNameY)
		if(dataset == "POStagSentence"):
			numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = SANItf2_loadDataset.loadDatasetType3(datasetType1FileNameX, generatePOSunambiguousInput, onlyAddPOSunambiguousInputToTrain, useSmallSentenceLengths)
		elif(dataset == "NewThyroid"):
			datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = SANItf2_loadDataset.loadDatasetType2(datasetType2FileName)

		shuffleSize = datasetNumExamples	#10*batchSize
		trainData = tf.data.Dataset.from_tensor_slices((train_x, train_y))
		trainData = trainData.repeat().shuffle(shuffleSize).batch(batchSize).prefetch(1)	#do not repeat

		for batchIndex, (batchX, batchY) in enumerate(trainData.take(trainingSteps), 1):

			if(algorithm == "ANN"):
				executeOptimisation(batchX, batchY)

				if(batchIndex % displayStep == 0):
					pred = neuralNetworkPropagation(batchX)
					loss = crossEntropy(pred, batchY)
					acc = calculateAccuracy(pred, batchY)
					print("batchIndex: %i, loss: %f, accuracy: %f" % (batchIndex, loss, acc))
			else:
				#learning algorithm not yet implemented:
				pred = neuralNetworkPropagation(batchX)
				
				if(batchIndex % displayStep == 0):
					pred = neuralNetworkPropagation(batchX)
					acc = tf.reduce_mean(tf.dtypes.cast(pred, tf.float32))
					print("batchIndex: %i, accuracy: %f" % (batchIndex, acc))

		if(algorithm == "ANN"):
			pred = neuralNetworkPropagation(test_x)
			print("Test Accuracy: %f" % calculateAccuracy(pred, test_y))
		else:
			#learning algorithm not yet implemented:
			pythonDummy = 1
