# -*- coding: utf-8 -*-
"""ANNtf2.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Usage:
python3 ANNtf2.py

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

from ANNtf2_operations import *
import ANNtf2_globalDefs
from numpy import random

algorithm = "ANN"
#algorithm = "SUANN"
#algorithm = "AUANN"
#algorithm = "HUANN"
#algorithm = "CUANN"
#algorithm = "SANI"

costCrossEntropyWithLogits = False
if(algorithm == "SANI"):
	algorithmSANI = "sharedModulesBinary"
	#algorithmSANI = "sharedModules"
	#algorithmSANI = "repeatedModules"
	if(algorithmSANI == "repeatedModules"):
		import ANNtf2_algorithmSANIrepeatedModules as ANNtf2_algorithmSANI
	elif(algorithmSANI == "sharedModules"):
		import ANNtf2_algorithmSANIsharedModules as ANNtf2_algorithmSANI
		costCrossEntropyWithLogits = True
	elif(algorithmSANI == "sharedModulesBinary"):
		import ANNtf2_algorithmSANIsharedModulesBinary as ANNtf2_algorithmSANI
elif(algorithm == "ANN"):
	import ANNtf2_algorithmANN
elif(algorithm == "HUANN"):
	import ANNtf2_algorithmHUANN
elif(algorithm == "SUANN"):
	import ANNtf2_algorithmSUANN
elif(algorithm == "AUANN"):
	import ANNtf2_algorithmAUANN
elif(algorithm == "CUANN"):
	import ANNtf2_algorithmCUANN
		
import ANNtf2_loadDataset

#learningRate, trainingSteps, batchSize, displayStep, numEpochs = -1

#performance enhancements for development environment only: 
debugUseSmallPOStagSequenceDataset = True	#def:False	#switch increases performance during development	#eg data-POStagSentence-smallBackup
useSmallSentenceLengths = True	#def:False	#switch increases performance during development	#eg data-simple-POStagSentence-smallBackup
trainMultipleFiles = False	#def:True	#switch increases performance during development	#eg data-POStagSentence
trainMultipleNetworks = False	#improve classification accuracy by averaging over multiple independently trained networks (test)
if(trainMultipleNetworks):
	numberOfNetworks = 5
else:
	numberOfNetworks = 1
	
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
		if(ANNtf2_algorithmSANI.allowMultipleContributingSubinputsPerSequentialInput):
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
	dataset = "SmallDataset"
	#trainMultipleNetworks = True	#default: False
	#numberOfNetworks = 3	#default: 1
elif(algorithm == "HUANN"):
	#dataset = "POStagSequence"
	dataset = "SmallDataset"
	#trainMultipleNetworks = True	#default: False
	#numberOfNetworks = 5	#default: 1
	trainHebbianBackprop = False	#default: False
elif(algorithm == "SUANN"):
	#dataset = "POStagSequence"
	dataset = "SmallDataset"
	#trainMultipleNetworks = True	#default: False
	#numberOfNetworks = 5	#default: 1
elif(algorithm == "AUANN"):
	#dataset = "POStagSequence"
	dataset = "SmallDataset"
	#trainMultipleNetworks = True	#default: False
	#numberOfNetworks = 5	#default: 1
elif(algorithm == "CUANN"):
	#dataset = "POStagSequence"
	dataset = "SmallDataset"
	#trainMultipleNetworks = True	#default: False
	#numberOfNetworks = 5	#default: 1

if(dataset == "SmallDataset"):
	#trainMultipleFiles = False	#required
	smallDatasetDefinitionsHeader = {'index':0, 'name':1, 'fileName':2, 'classColumnFirst':3}	
	smallDatasetDefinitions = [
	(0, "New Thyroid", "new-thyroid.data", True),
	(1, "Swedish Auto Insurance", "UNAVAILABLE.txt", False),	#AutoInsurSweden.txt BAD
	(2, "Wine Quality Dataset", "winequality-whiteFormatted.csv", False),
	(3, "Pima Indians Diabetes Dataset", "pima-indians-diabetes.csv", False),
	(4, "Sonar Dataset", "sonar.all-data", False),
	(5, "Banknote Dataset", "data_banknote_authentication.txt", False),
	(6, "Iris Flowers Dataset", "iris.data", False),
	(7, "Abalone Dataset", "UNAVAILABLE", False),	#abaloneFormatted.data BAD
	(8, "Ionosphere Dataset", "ionosphere.data", False),
	(9, "Wheat Seeds Dataset", "seeds_datasetFormatted.txt", False),
	(10, "Boston House Price Dataset", "UNAVAILABLE", False)	#housingFormatted.data BAD
	]
	smallDatasetIndex = 0 #default: 0 (New Thyroid)
	datasetFileName = smallDatasetDefinitions[smallDatasetIndex][smallDatasetDefinitionsHeader['fileName']]
	datasetClassColumnFirst = smallDatasetDefinitions[smallDatasetIndex][smallDatasetDefinitionsHeader['classColumnFirst']]
	print("datasetFileName = ", datasetFileName)
	print("datasetClassColumnFirst = ", datasetClassColumnFirst)
			
if(debugUseSmallPOStagSequenceDataset):
	datasetFileNameXstart = "XdatasetPartSmall"
	datasetFileNameYstart = "YdatasetPartSmall"
else:
	datasetFileNameXstart = "XdatasetPart"
	datasetFileNameYstart = "YdatasetPart"
datasetFileNameXend = ".dat"
datasetFileNameYend = ".dat"
datasetFileNameStart = "datasetPart"
datasetFileNameEnd = ".dat"


	
	
def neuralNetworkPropagation(x, networkIndex=1):
	if(algorithm == "SANI"):
		pred = ANNtf2_algorithmSANI.neuralNetworkPropagationSANI(x)
	elif(algorithm == "ANN"):
		pred = ANNtf2_algorithmANN.neuralNetworkPropagationANN(x, networkIndex)
	elif(algorithm == "HUANN"):
		pred = ANNtf2_algorithmHUANN.neuralNetworkPropagationHUANN(x, networkIndex)
	elif(algorithm == "SUANN"):
		pred = ANNtf2_algorithmSUANN.neuralNetworkPropagationSUANN(x, networkIndex)
	elif(algorithm == "AUANN"):
		pred = ANNtf2_algorithmAUANN.neuralNetworkPropagationAUANN(x, networkIndex)
	elif(algorithm == "CUANN"):
		pred = ANNtf2_algorithmCUANN.neuralNetworkPropagationCUANN(x, networkIndex)
	return pred
	

def executeLearning(x, y, networkIndex=1):
	if(algorithm == "HUANN"):
		#learning algorithm embedded in forward propagation
		if(trainHebbianBackprop):
			pred = ANNtf2_algorithmHUANN.neuralNetworkPropagationHUANNtrain(x, y, networkIndex, trainHebbianBackprop=True, trainHebbianLastLayerSupervision=True)
		else:
			pred = ANNtf2_algorithmHUANN.neuralNetworkPropagationHUANNtrain(x, y, networkIndex, trainHebbianForwardprop=True, trainHebbianLastLayerSupervision=True)
	elif(algorithm == "SUANN"):
		#learning algorithm embedded in multiple iterations of forward propagation
		pred = ANNtf2_algorithmSUANN.neuralNetworkPropagationSUANNtrain(x, y, networkIndex)

def executeLearningAUANN(x, y, exemplarsX, exemplarsY, currentClassTarget, networkIndex=1):
	#learning algorithm embedded in forward propagation of new class x experience following forward propagation of existing class x experience
	pred = ANNtf2_algorithmAUANN.neuralNetworkPropagationAUANNtrain(x, y, exemplarsX, exemplarsY, currentClassTarget, networkIndex)

def executeLearningCUANN(x, y, networkIndex=1):	#currentClassTarget
	#learning algorithm embedded in forward propagation of new class x experience following forward propagation of existing class x experience
	pred = ANNtf2_algorithmCUANN.neuralNetworkPropagationCUANNtrain(x, y, networkIndex)

	
			
def executeOptimisation(x, y, networkIndex=1):
	with tf.GradientTape() as g:
		pred = neuralNetworkPropagation(x, networkIndex)
		loss = crossEntropy(pred, y, datasetNumClasses, costCrossEntropyWithLogits)
		
	if(algorithm == "SANI"):
		if(algorithmSANI == "sharedModules"):
			if(ANNtf2_algorithmSANI.useSparseTensors):
				if(ANNtf2_algorithmSANI.performSummationOfSubInputsWeighted):
					if(ANNtf2_algorithmSANI.performSummationOfSequentialInputsWeighted):
						trainableVariables = list(ANNtf2_algorithmSANI.W.values()) + list(ANNtf2_algorithmSANI.Wseq.values())	#+ list(ANNtf2_algorithmSANI.B.values()) 
					else:
						trainableVariables = list(ANNtf2_algorithmSANI.Wseq.values())
				else:
					if(ANNtf2_algorithmSANI.performSummationOfSequentialInputsWeighted):
						trainableVariables = list(ANNtf2_algorithmSANI.W.values()) #+ list(ANNtf2_algorithmSANI.B.values())
					else:
						trainableVariables = list()	
			else:
				if(ANNtf2_algorithmSANI.performSummationOfSequentialInputsWeighted):
					trainableVariables = list(ANNtf2_algorithmSANI.W.values()) + list(ANNtf2_algorithmSANI.Wseq.values()) + list(ANNtf2_algorithmSANI.Bseq.values())	#+ list(ANNtf2_algorithmSANI.B.values()) 
				else:
					trainableVariables = list(ANNtf2_algorithmSANI.Wseq.values()) + list(ANNtf2_algorithmSANI.Bseq.values())
					#trainableVariables = list(ANNtf2_algorithmSANI.Wseq.values())
					
		elif(algorithmSANI == "repeatedModules"):
			if(ANNtf2_algorithmSANI.allowMultipleSubinputsPerSequentialInput):
				if(ANNtf2_algorithmSANI.performSummationOfSequentialInputsWeighted):
					if(ANNtf2_algorithmSANI.performSummationOfSubInputsWeighted):
						trainableVariables = list(ANNtf2_algorithmSANI.W.values()) + list(ANNtf2_algorithmSANI.Wseq.values())
						#trainableVariables = list(ANNtf2_algorithmSANI.W.values()) + list(ANNtf2_algorithmSANI.B.values()) + list(ANNtf2_algorithmSANI.Wseq.values()) + list(ANNtf2_algorithmSANI.Bseq.values())
					else:
						trainableVariables = list(ANNtf2_algorithmSANI.W.values())
						#trainableVariables = list(ANNtf2_algorithmSANI.W.values()) + list(ANNtf2_algorithmSANI.B.values())
				else:
					if(ANNtf2_algorithmSANI.performSummationOfSubInputsWeighted):
						trainableVariables = list(ANNtf2_algorithmSANI.Wseq.values())
						#trainableVariables = list(ANNtf2_algorithmSANI.Wseq.values()) + list(ANNtf2_algorithmSANI.Bseq.values())
					else:
						print("error: allowMultipleSubinputsPerSequentialInput && !performSummationOfSequentialInputsWeighted && !performSummationOfSubInputsWeighted")
			else:
				if(ANNtf2_algorithmSANI.performSummationOfSequentialInputsWeighted):
					trainableVariables = list(ANNtf2_algorithmSANI.W.values())
					#trainableVariables = list(ANNtf2_algorithmSANI.W.values()) + list(ANNtf2_algorithmSANI.B.values())
				else:
					print("error: !allowMultipleSubinputsPerSequentialInput && !performSummationOfSequentialInputsWeighted")
	elif(algorithm == "ANN"):
		Wlist = []
		Blist = []
		for l in range(1, numberOfLayers+1):
			Wlist.append(ANNtf2_algorithmANN.W[generateParameterNameNetwork(networkIndex, l, "W")])
			Blist.append(ANNtf2_algorithmANN.B[generateParameterNameNetwork(networkIndex, l, "B")])
		trainableVariables = Wlist + Blist

	gradients = g.gradient(loss, trainableVariables)
	
	optimizer.apply_gradients(zip(gradients, trainableVariables))

	

																
if __name__ == "__main__":

	#generate network parameters based on dataset properties:

	#global learningRate, trainingSteps, batchSize, displayStep, numEpochs
	
	datasetNumExamplesTemp = 0
	datasetNumFeatures = 0
	datasetNumClasses = 0

	fileIndexTemp = 0
	fileIndexStr = str(fileIndexTemp).zfill(4)
	if(dataset == "POStagSequence"):
		datasetType1FileNameX = datasetFileNameXstart + fileIndexStr + datasetFileNameXend
		datasetType1FileNameY = datasetFileNameYstart + fileIndexStr + datasetFileNameYend
	elif(dataset == "SmallDataset"):
		if(trainMultipleFiles):
			datasetType2FileName = datasetFileNameStart + fileIndexStr + datasetFileNameEnd
		else:
			datasetType2FileName = datasetFileName

	numberOfLayers = 0
	if(dataset == "POStagSequence"):
		datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType1(datasetType1FileNameX, datasetType1FileNameY)
	elif(dataset == "POStagSentence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType3(datasetType1FileNameX, generatePOSunambiguousInput, onlyAddPOSunambiguousInputToTrain, useSmallSentenceLengths)
	elif(dataset == "SmallDataset"):
		datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType2(datasetType2FileName, datasetClassColumnFirst)


	#Model constants
	num_input_neurons = datasetNumFeatures  #train_x.shape[1]
	num_output_neurons = datasetNumClasses
	if(algorithm == "AUANN"):
		num_output_neurons = ANNtf2_algorithmAUANN.calculateOutputNeuronsAUANN(datasetNumClasses)

	if(algorithm == "SANI"):
		if(dataset == "POStagSentence"):
			ANNtf2_algorithmSANI.defineTrainingParametersSANIsharedModules(numberOfFeaturesPerWord, paddingTagIndex)
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = ANNtf2_algorithmSANI.defineTrainingParametersSANI(dataset, trainMultipleFiles)
		ANNtf2_algorithmSANI.defineNetworkParametersSANI(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, useSmallSentenceLengths)
		ANNtf2_algorithmSANI.defineNeuralNetworkParametersSANI()
	elif(algorithm == "ANN"):
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = ANNtf2_algorithmANN.defineTrainingParametersANN(dataset, trainMultipleFiles)
		numberOfLayers = ANNtf2_algorithmANN.defineNetworkParametersANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworks)
		ANNtf2_algorithmANN.defineNeuralNetworkParametersANN()
	elif(algorithm == "HUANN"):
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = ANNtf2_algorithmHUANN.defineTrainingParametersHUANN(dataset, trainMultipleFiles)
		numberOfLayers = ANNtf2_algorithmHUANN.defineNetworkParametersHUANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworks)
		ANNtf2_algorithmHUANN.defineNeuralNetworkParametersHUANN()
	elif(algorithm == "SUANN"):
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = ANNtf2_algorithmSUANN.defineTrainingParametersSUANN(dataset, trainMultipleFiles)
		numberOfLayers = ANNtf2_algorithmSUANN.defineNetworkParametersSUANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworks)
		ANNtf2_algorithmSUANN.defineNeuralNetworkParametersSUANN()		
	elif(algorithm == "AUANN"):
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = ANNtf2_algorithmAUANN.defineTrainingParametersAUANN(dataset, trainMultipleFiles)
		numberOfLayers = ANNtf2_algorithmAUANN.defineNetworkParametersAUANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworks)
		ANNtf2_algorithmAUANN.defineNeuralNetworkParametersAUANN()	
	elif(algorithm == "CUANN"):
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = ANNtf2_algorithmCUANN.defineTrainingParametersCUANN(dataset, trainMultipleFiles)
		numberOfLayers = ANNtf2_algorithmCUANN.defineNetworkParametersCUANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworks)
		ANNtf2_algorithmCUANN.defineNeuralNetworkParametersCUANN()	
		
	#define epochs:

	if(trainMultipleFiles):
		fileIndexFirst = 0
		if(useSmallSentenceLengths):
			fileIndexLast = 11
		else:
			fileIndexLast = 1202

	noisySampleGeneration = False
	if(algorithm == "SUANN"):
		noisySampleGeneration, noisySampleGenerationNumSamples, noiseStandardDeviation = ANNtf2_algorithmSUANN.getNoisySampleGenerationNumSamples()
	if(algorithm == "AUANN"):
		noisySampleGeneration, noisySampleGenerationNumSamples, noiseStandardDeviation = ANNtf2_algorithmAUANN.getNoisySampleGenerationNumSamples()
	if(algorithm == "CUANN"):
		noisySampleGeneration, noisySampleGenerationNumSamples, noiseStandardDeviation = ANNtf2_algorithmCUANN.getNoisySampleGenerationNumSamples()
	if(noisySampleGeneration):
		batchXmultiples = tf.constant([noisySampleGenerationNumSamples, 1], tf.int32)
		batchYmultiples = tf.constant([noisySampleGenerationNumSamples], tf.int32)
		randomNormal = tf.initializers.RandomNormal()	#tf.initializers.RandomUniform(minval=-1, maxval=1)

	# Stochastic gradient descent optimizer.
	optimizer = tf.optimizers.SGD(learningRate)

	for e in range(numEpochs):

		print("epoch e = ", e)

		if(trainMultipleFiles):
			fileIndexArray = np.arange(fileIndexFirst, fileIndexLast+1, 1)
			#print("fileIndexArray = " + str(fileIndexArray))
			np.random.shuffle(fileIndexArray)
			fileIndexShuffledArray = fileIndexArray
			#print("fileIndexShuffledArray = " + str(fileIndexShuffledArray))
		else:
			fileIndexShuffledArray = [0]
			
		for fileIndex in fileIndexShuffledArray:	#range(fileIndexFirst, fileIndexLast+1):

			#print("fileIndex = ", fileIndex)
			
			datasetNumExamples = 0

			fileIndexStr = str(fileIndex).zfill(4)
			if(dataset == "POStagSequence"):
				datasetType1FileNameX = datasetFileNameXstart + fileIndexStr + datasetFileNameXend
				datasetType1FileNameY = datasetFileNameYstart + fileIndexStr + datasetFileNameYend
			elif(dataset == "SmallDataset"):
				if(trainMultipleFiles):
					datasetType2FileName = datasetFileNameStart + fileIndexStr + datasetFileNameEnd
				else:
					datasetType2FileName = datasetFileName

			if(dataset == "POStagSequence"):
				datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = ANNtf2_loadDataset.loadDatasetType1(datasetType1FileNameX, datasetType1FileNameY)
			if(dataset == "POStagSentence"):
				numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = ANNtf2_loadDataset.loadDatasetType3(datasetType1FileNameX, generatePOSunambiguousInput, onlyAddPOSunambiguousInputToTrain, useSmallSentenceLengths)
			elif(dataset == "SmallDataset"):
				datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = ANNtf2_loadDataset.loadDatasetType2(datasetType2FileName, datasetClassColumnFirst)

			shuffleSize = datasetNumExamples	#heuristic: 10*batchSize
			
			#new iteration method (only required for algorithm == "AUANN/CUANN"):	
			datasetNumClassesActual = datasetNumClasses
			trainDataIndex = 0
			if(algorithm == "AUANN"):
				currentClassTarget = 0
				generateClassTargetExemplars = False
				if(e == 0):
					generateClassTargetExemplars = True
				networkIndex = 1 #note ANNtf2_algorithmAUANN doesn't currently support multiple networks
				trainDataList = ANNtf2_algorithmAUANN.generateTFtrainDataFromNParraysAUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses)
				exemplarDataList = ANNtf2_algorithmAUANN.generateTFexemplarDataFromNParraysAUANN(train_x, train_y, networkIndex, shuffleSize, batchSize, datasetNumClasses, generateClassTargetExemplars)
				test_y = ANNtf2_algorithmAUANN.generateYActualfromYAUANN(test_y, num_output_neurons)
				datasetNumClassTargets = datasetNumClasses
				datasetNumClasses = ANNtf2_algorithmAUANN.generateNumClassesActualAUANN(datasetNumClasses, num_output_neurons)
				trainDataListIterators = []
				for trainData in trainDataList:
					trainDataListIterators.append(iter(trainData))
				exemplarDataListIterators = []
				for exemplarData in exemplarDataList:
					exemplarDataListIterators.append(iter(exemplarData))
			elif(algorithm == "CUANN"):
				trainDataList = ANNtf2_algorithmCUANN.generateTFtrainDataFromNParraysCUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses)
				trainDataListIterators = []
				for trainData in trainDataList:
					trainDataListIterators.append(iter(trainData))
			else:
				trainData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)
				trainDataList = []
				trainDataList.append(trainData)
				trainDataListIterators = []
				for trainData in trainDataList:
					trainDataListIterators.append(iter(trainData))
	
			#original iteration method:
			#trainData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize):	
			#for batchIndex, (batchX, batchY) in enumerate(trainData.take(trainingSteps), 1):	
					
			#new iteration method:			
			#print("trainingSteps = ", trainingSteps)
			#print("batchSize = ", batchSize)
			
			for batchIndex in range(int(trainingSteps*shuffleSize*10/batchSize)):		#*5 is to normalise number of final training steps new iteration method in relative to original iteration method
				(batchX, batchY) = trainDataListIterators[trainDataIndex].get_next()	#next(trainDataListIterators[trainDataIndex])
				
				batchYActual = batchY
				if(algorithm == "AUANN"):
					(exemplarsX, exemplarsY) = exemplarDataListIterators[trainDataIndex].get_next()
					batchYactual = ANNtf2_algorithmAUANN.generateTFYActualfromYandExemplarYAUANN(batchY, exemplarsY)
								
				if(noisySampleGeneration):
					if(batchSize != 1):	#batchX.shape[0]
						print("error: noisySampleGeneration && batchSize != 1")
						exit()
					batchX = tf.tile(batchX, batchXmultiples)
					batchY = tf.tile(batchY, batchYmultiples)
					batchXnoise = tf.math.multiply(tf.constant(randomNormal(batchX.shape), tf.float32), noiseStandardDeviation)
					batchX = tf.math.add(batchX, batchXnoise)
					#print("batchX = ", batchX)
					#print("batchY = ", batchY)

				predNetworkAverage = tf.Variable(tf.zeros(datasetNumClasses))

				#print("datasetNumClasses = ", datasetNumClasses)
				#print("batchX.shape = ", batchX.shape)
				#print("batchY.shape = ", batchY.shape)
				
				for networkIndex in range(1, numberOfNetworks+1):

					if(algorithm == "ANN"):
						executeOptimisation(batchX, batchY, networkIndex)
						if(batchIndex % displayStep == 0):
							pred = neuralNetworkPropagation(batchX, networkIndex)
							#print("pred.shape = ", pred.shape)
							loss = crossEntropy(pred, batchY, datasetNumClasses, costCrossEntropyWithLogits)
							acc = calculateAccuracy(pred, batchY)
							print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
							predNetworkAverage = predNetworkAverage + pred
					elif(algorithm == "HUANN"):
						batchYoneHot = tf.one_hot(batchY, depth=datasetNumClasses)
						executeLearning(batchX, batchYoneHot, networkIndex)
						if(batchIndex % displayStep == 0):
							pred = neuralNetworkPropagation(batchX, networkIndex)
							loss = crossEntropy(pred, batchY, datasetNumClasses, costCrossEntropyWithLogits)
							acc = calculateAccuracy(pred, batchY)
							print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
							predNetworkAverage = predNetworkAverage + pred
					elif(algorithm == "SUANN"):
						executeLearning(batchX, batchY, networkIndex)
						if(batchIndex % displayStep == 0):
							pred = neuralNetworkPropagation(batchX, networkIndex)
							loss = crossEntropy(pred, batchY, datasetNumClasses, costCrossEntropyWithLogits)
							acc = calculateAccuracy(pred, batchY)
							print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
							predNetworkAverage = predNetworkAverage + pred
					elif(algorithm == "AUANN"):
						#learning algorithm embedded in forward propagation of new class x experience following forward propagation of existing class x experience
						executeLearningAUANN(batchX, batchY, exemplarsX, exemplarsY, currentClassTarget, networkIndex)
						if(batchIndex % displayStep == 0):
							pred = neuralNetworkPropagation(batchX, networkIndex)
							loss = crossEntropy(pred, batchYactual, datasetNumClasses, costCrossEntropyWithLogits)
							acc = calculateAccuracy(pred, batchYactual)
							print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
							predNetworkAverage = predNetworkAverage + pred
					elif(algorithm == "CUANN"):
						executeLearningCUANN(batchX, batchY, networkIndex)	#currentClassTarget
						if(batchIndex % displayStep == 0):
							pred = neuralNetworkPropagation(batchX, networkIndex)
							loss = crossEntropy(pred, batchY, datasetNumClasses, costCrossEntropyWithLogits)
							acc = calculateAccuracy(pred, batchY)
							print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
							predNetworkAverage = predNetworkAverage + pred
					elif(algorithm == "SANI"):
						#learning algorithm not yet implemented:
						if(batchSize > 1):
							pred = neuralNetworkPropagation(batchX)
						if(batchIndex % displayStep == 0):
							pred = neuralNetworkPropagation(batchX)
							acc = tf.reduce_mean(tf.dtypes.cast(pred, tf.float32))
							print("batchIndex: %i, accuracy: %f" % (batchIndex, acc))


				if(algorithm == "AUANN"):
					batchY = ANNtf2_algorithmAUANN.generateTFYActualfromYandExemplarYAUANN(batchY, exemplarsY)
					currentClassTarget = currentClassTarget+1
					if(currentClassTarget == datasetNumClassTargets):
						currentClassTarget = 0
					trainDataIndex = currentClassTarget
							
				if(batchIndex % displayStep == 0):
					if(trainMultipleNetworks):
						predNetworkAverage = predNetworkAverage / numberOfNetworks
						loss = crossEntropy(predNetworkAverage, batchYActual, datasetNumClasses, costCrossEntropyWithLogits)
						acc = calculateAccuracy(predNetworkAverage, batchYActual)
						print("batchIndex: %i, loss: %f, accuracy: %f" % (batchIndex, loss, acc))	



			predNetworkAverageAll = tf.Variable(tf.zeros([test_y.shape[0], datasetNumClasses]))
			for networkIndex in range(1, numberOfNetworks+1):
				if(algorithm == "ANN"):
					pred = neuralNetworkPropagation(test_x, networkIndex)	#test_x batch may be too large to propagate simultaneously and require subdivision
					print("Test Accuracy: networkIndex: %i, %f" % (networkIndex, calculateAccuracy(pred, test_y)))
					predNetworkAverageAll = predNetworkAverageAll + pred
				elif(algorithm == "HUANN"):
					pred = neuralNetworkPropagation(test_x, networkIndex)
					print("Test Accuracy: networkIndex: %i, %f" % (networkIndex, calculateAccuracy(pred, test_y)))
					predNetworkAverageAll = predNetworkAverageAll + pred
				elif(algorithm == "SUANN"):
					pred = neuralNetworkPropagation(test_x, networkIndex)
					print("Test Accuracy: networkIndex: %i, %f" % (networkIndex, calculateAccuracy(pred, test_y)))
					predNetworkAverageAll = predNetworkAverageAll + pred
				elif(algorithm == "AUANN"):
					pred = neuralNetworkPropagation(test_x, networkIndex)
					print("Test Accuracy: networkIndex: %i, %f" % (networkIndex, calculateAccuracy(pred, test_y)))
					predNetworkAverageAll = predNetworkAverageAll + pred
				elif(algorithm == "CUANN"):
					pred = neuralNetworkPropagation(test_x, networkIndex)
					print("Test Accuracy: networkIndex: %i, %f" % (networkIndex, calculateAccuracy(pred, test_y)))
					predNetworkAverageAll = predNetworkAverageAll + pred
				elif(algorithm == "SANI"):
					#learning algorithm not yet implemented:
					pythonDummy = 1

			if(trainMultipleNetworks):
					predNetworkAverageAll = predNetworkAverageAll / numberOfNetworks
					#print("predNetworkAverageAll", predNetworkAverageAll)
					acc = calculateAccuracy(predNetworkAverageAll, test_y)
					print("Test Accuracy: %f" % (acc))
