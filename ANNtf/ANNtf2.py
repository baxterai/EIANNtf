# -*- coding: utf-8 -*-
"""ANNtf2.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Usage:
python3 ANNtf2.py

# Description

Train an artificial neural network (ANN or SANI or CANN)

- Author: Richard Bruce Baxter - Copyright (c) 2020-2021 Baxter AI (baxterai.com)

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
import ANNtf2_loadDataset

#algorithm = "ANN"
#algorithm = "SANI"
algorithm = "CANN"
#algorithm = "FBANN"

suppressGradientDoNotExistForVariablesWarnings = True

costCrossEntropyWithLogits = False
if(algorithm == "ANN"):
	import ANNtf2_algorithmANN
elif(algorithm == "SANI"):
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
elif(algorithm == "CANN"):
	#algorithmCANN = "CANN_expHUANN"
	#algorithmCANN = "CANN_expSUANN"
	#algorithmCANN = "CANN_expAUANN"
	#algorithmCANN = "CANN_expCUANN"
	#algorithmCANN = "CANN_expXUANN"
	#algorithmCANN = "CANN_expMUANN"
	algorithmCANN = "CANN_expRUANN"
	if(algorithmCANN == "CANN_expHUANN"):
		import ANNtf2_algorithmCANN_expHUANN as ANNtf2_algorithmCANN
	elif(algorithmCANN == "CANN_expSUANN"):
		import ANNtf2_algorithmCANN_expSUANN as ANNtf2_algorithmCANN
	elif(algorithmCANN == "CANN_expAUANN"):
		import ANNtf2_algorithmCANN_expAUANN as ANNtf2_algorithmCANN
	elif(algorithmCANN == "CANN_expCUANN"):
		import ANNtf2_algorithmCANN_expCUANN as ANNtf2_algorithmCANN
	elif(algorithmCANN == "CANN_expXUANN"):
		XUANNnegativeSamplesComplement = False	#default: True
		XUANNnegativeSamplesAll = False	#default: False #orig implementation
		XUANNnegativeSamplesRandom = True	#default: False 
		import ANNtf2_algorithmCANN_expXUANN as ANNtf2_algorithmCANN
	elif(algorithmCANN == "CANN_expMUANN"):
		import ANNtf2_algorithmCANN_expMUANN as ANNtf2_algorithmCANN		
	elif(algorithmCANN == "CANN_expRUANN"):
		import ANNtf2_algorithmCANN_expRUANN as ANNtf2_algorithmCANN
elif(algorithm == "FBANN"):
	import ANNtf2_algorithmFBANN as ANNtf2_algorithmFBANN
				
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
elif(algorithm == "CANN"):
	#dataset = "POStagSequence"
	dataset = "SmallDataset"
	#trainMultipleNetworks = True	#default: False
	#numberOfNetworks = 5	#default: 1
	trainHebbianBackprop = False	#default: False
elif(algorithm == "FBANN"):
	#dataset = "POStagSequence"
	dataset = "SmallDataset"
	#trainMultipleNetworks = True	#default: False
	#numberOfNetworks = 5	#default: 1
		
	
if(dataset == "SmallDataset"):
	smallDatasetIndex = 0 #default: 0 (New Thyroid)
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
	elif(algorithm == "CANN"):
		pred = ANNtf2_algorithmCANN.neuralNetworkPropagationCANN(x, networkIndex)
	elif(algorithm == "FBANN"):
		pred = ANNtf2_algorithmFBANN.neuralNetworkPropagationFBANNwrapper(x, networkIndex)
	return pred
	

#algorithm CANN:
def executeLearningCANN(x, y, networkIndex=1):
	if(algorithmCANN == "CANN_expHUANN"):
		#learning algorithm embedded in forward propagation
		if(trainHebbianBackprop):
			pred = ANNtf2_algorithmCANN.neuralNetworkPropagationCANN_expHUANNtrain(x, y, networkIndex, trainHebbianBackprop=True, trainHebbianLastLayerSupervision=True)
		else:
			pred = ANNtf2_algorithmCANN.neuralNetworkPropagationCANN_expHUANNtrain(x, y, networkIndex, trainHebbianForwardprop=True, trainHebbianLastLayerSupervision=True)
	elif(algorithmCANN == "CANN_expSUANN"):
		#learning algorithm embedded in multiple iterations of forward propagation
		pred = ANNtf2_algorithmCANN.neuralNetworkPropagationCANN_expSUANNtrain(x, y, networkIndex)
	elif(algorithmCANN == "CANN_expCUANN"):
		#learning algorithm embedded in forward propagation of new class x experience following forward propagation of existing class x experience
		pred = ANNtf2_algorithmCANN.neuralNetworkPropagationCANN_expCUANNtrain(x, y, networkIndex)
	elif(algorithmCANN == "CANN_expMUANN"):
		#learning algorithm embedded in multiple forward propagation and synaptic delta calculations
		pred = ANNtf2_algorithmCANN.neuralNetworkPropagationCANN_expMUANNtrain(x, y, networkIndex)
	elif(algorithmCANN == "CANN_expRUANN"):
		#learning algorithm: in reverse order, stocastically establishing Aideal of each layer (by temporarily biasing firing rate of neurons) to better achieve Aideal of higher layer (through multiple local/single layer forward propagations), then (simultaneous/parallel layer processing) stocastically adjusting weights to fine tune towards Aideal of their higher layers
		pred = ANNtf2_algorithmCANN.neuralNetworkPropagationCANN_expRUANNtrain(x, y, networkIndex)
def executeLearningCANN_expAUANN(x, y, exemplarsX, exemplarsY, currentClassTarget, networkIndex=1):
	#learning algorithm embedded in forward propagation of new class x experience following forward propagation of existing class x experience
	pred = ANNtf2_algorithmCANN.neuralNetworkPropagationCANN_expAUANNtrain(x, y, exemplarsX, exemplarsY, currentClassTarget, networkIndex)
def executeLearningCANN_expXUANN(x, y, samplePositiveX, samplePositiveY, sampleNegativeX, sampleNegativeY, networkIndex):
	#learning algorithm: perform contrast training (diff of interclass experience with current experience, and diff of extraclass experience with current experience) at each layer of network
	pred = ANNtf2_algorithmCANN.neuralNetworkPropagationCANN_expXUANNtrain(x, y, samplePositiveX, samplePositiveY, sampleNegativeX, sampleNegativeY, networkIndex)


			
def executeOptimisation(x, y, networkIndex=1):
	with tf.GradientTape() as g:
		pred = neuralNetworkPropagation(x, networkIndex)
		loss = crossEntropy(pred, y, datasetNumClasses, costCrossEntropyWithLogits)
		
	if(algorithm == "ANN"):
		Wlist = []
		Blist = []
		for l in range(1, numberOfLayers+1):
			if(ANNtf2_algorithmANN.debugOnlyTrainFinalLayer):
				if(l == numberOfLayers):
					Wlist.append(ANNtf2_algorithmANN.W[generateParameterNameNetwork(networkIndex, l, "W")])
					Blist.append(ANNtf2_algorithmANN.B[generateParameterNameNetwork(networkIndex, l, "B")])				
			else:	
				Wlist.append(ANNtf2_algorithmANN.W[generateParameterNameNetwork(networkIndex, l, "W")])
				Blist.append(ANNtf2_algorithmANN.B[generateParameterNameNetwork(networkIndex, l, "B")])
		trainableVariables = Wlist + Blist
	elif(algorithm == "SANI"):
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
	elif(algorithm == "FBANN"):
		Wflist = []
		Wblist = []
		Blist = []
		for l1 in range(1, ANNtf2_algorithmFBANN.highestLayer+1):
			if(ANNtf2_algorithmFBANN.supportSkipLayers):
				for l2 in range(0, l1):
					if(l2 < l1):
						Wflist.append(ANNtf2_algorithmFBANN.Wf[generateParameterNameNetworkSkipLayers(networkIndex, l1, l2, "Wf")])
				if(ANNtf2_algorithmFBANN.feedbackConnections):
					if((l1 <= ANNtf2_algorithmFBANN.highestLayerWithIncomingBackwardsConnections) and (l1 >= ANNtf2_algorithmFBANN.lowestLayerWithIncomingBackwardsConnections)):
						for l2 in range(l1+1, ANNtf2_algorithmFBANN.highestLayer+1):
							if(l2 > l1):
								Wblist.append(ANNtf2_algorithmFBANN.Wb[generateParameterNameNetworkSkipLayers(networkIndex, l1, l2, "Wb")])
			else:
				Wflist.append(ANNtf2_algorithmFBANN.Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")])
				if(ANNtf2_algorithmFBANN.feedbackConnections):
					if((l1 <= ANNtf2_algorithmFBANN.highestLayerWithIncomingBackwardsConnections) and (l1 >= ANNtf2_algorithmFBANN.lowestLayerWithIncomingBackwardsConnections)):
						Wblist.append(ANNtf2_algorithmFBANN.Wb[generateParameterNameNetwork(networkIndex, l1, "Wb")])
								
			Blist.append(ANNtf2_algorithmFBANN.B[generateParameterNameNetwork(networkIndex, l1, "B")])
			
		if(ANNtf2_algorithmFBANN.feedbackConnections):
			trainableVariables = Wflist + Wblist + Blist
		else:
			trainableVariables = Wflist + Blist
	elif(algorithm == "CANN"):
		print("executeOptimisation error: algorithm CANN not supported, use executeLearningCANN() instead")
		exit()

		
	gradients = g.gradient(loss, trainableVariables)
	
	if(suppressGradientDoNotExistForVariablesWarnings):
		optimizer.apply_gradients([
    		(grad, var) 
    		for (grad, var) in zip(gradients, trainableVariables) 
    		if grad is not None
			])
	else:
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
	if(algorithm == "CANN"):
		if(algorithmCANN == "CANN_expAUANN"):
			num_output_neurons = ANNtf2_algorithmCANN_expAUANN.calculateOutputNeuronsCANN_expAUANN(datasetNumClasses)

	if(algorithm == "ANN"):
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = ANNtf2_algorithmANN.defineTrainingParametersANN(dataset, trainMultipleFiles)
		numberOfLayers = ANNtf2_algorithmANN.defineNetworkParametersANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworks)
		ANNtf2_algorithmANN.defineNeuralNetworkParametersANN()
	elif(algorithm == "SANI"):
		if(dataset == "POStagSentence"):
			ANNtf2_algorithmSANI.defineTrainingParametersSANIsharedModules(numberOfFeaturesPerWord, paddingTagIndex)
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = ANNtf2_algorithmSANI.defineTrainingParametersSANI(dataset, trainMultipleFiles)
		ANNtf2_algorithmSANI.defineNetworkParametersSANI(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, useSmallSentenceLengths)
		ANNtf2_algorithmSANI.defineNeuralNetworkParametersSANI()
	elif(algorithm == "CANN"):
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = ANNtf2_algorithmCANN.defineTrainingParametersCANN(dataset, trainMultipleFiles)
		numberOfLayers = ANNtf2_algorithmCANN.defineNetworkParametersCANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworks)
		ANNtf2_algorithmCANN.defineNeuralNetworkParametersCANN()
	elif(algorithm == "FBANN"):
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = ANNtf2_algorithmFBANN.defineTrainingParametersFBANN(dataset, trainMultipleFiles)
		numberOfLayers = ANNtf2_algorithmFBANN.defineNetworkParametersFBANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworks)
		ANNtf2_algorithmFBANN.defineNeuralNetworkParametersFBANN()
					
	#define epochs:

	if(trainMultipleFiles):
		fileIndexFirst = 0
		if(useSmallSentenceLengths):
			fileIndexLast = 11
		else:
			fileIndexLast = 1202

	noisySampleGeneration = False
	if(algorithm == "CANN"):
		noisySampleGeneration, noisySampleGenerationNumSamples, noiseStandardDeviation = ANNtf2_algorithmCANN.getNoisySampleGenerationNumSamples()
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
			
			#new iteration method (only required for algorithm == "CANN_expAUANN/CANN_expCUANN"):	
			datasetNumClassesActual = datasetNumClasses
			trainDataIndex = 0
			if(algorithm == "CANN"):
				if(algorithmCANN == "CANN_expAUANN"):
					currentClassTarget = 0
					generateClassTargetExemplars = False
					if(e == 0):
						generateClassTargetExemplars = True
					networkIndex = 1 #note ANNtf2_algorithmCANN_expAUANN doesn't currently support multiple networks
					trainDataList = ANNtf2_algorithmCANN.generateTFtrainDataFromNParraysCANN_expAUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses)
					exemplarDataList = ANNtf2_algorithmCANN.generateTFexemplarDataFromNParraysCANN_expAUANN(train_x, train_y, networkIndex, shuffleSize, batchSize, datasetNumClasses, generateClassTargetExemplars)
					test_y = ANNtf2_algorithmCANN.generateYActualfromYCANN_expAUANN(test_y, num_output_neurons)
					datasetNumClassTargets = datasetNumClasses
					datasetNumClasses = ANNtf2_algorithmCANN.generateNumClassesActualCANN_expAUANN(datasetNumClasses, num_output_neurons)
					exemplarDataListIterators = []
					for exemplarData in exemplarDataList:
						exemplarDataListIterators.append(iter(exemplarData))
				elif(algorithmCANN == "CANN_expCUANN"):
					trainDataList = ANNtf2_algorithmCANN.generateTFtrainDataFromNParraysCANN_expCUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses)
				elif(algorithmCANN == "CANN_expXUANN"):
					currentClassTarget = 0
					generateClassTargetExemplars = False
					if(e == 0):
						generateClassTargetExemplars = True
					trainDataList = ANNtf2_algorithmCANN.generateTFtrainDataFromNParraysCANN_expXUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses, generatePositiveSamples=True)
					datasetNumClassTargets = datasetNumClasses
					samplePositiveDataList = ANNtf2_algorithmCANN.generateTFtrainDataFromNParraysCANN_expXUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses, generatePositiveSamples=True)
					if(XUANNnegativeSamplesComplement):
						sampleNegativeDataList = ANNtf2_algorithmCANN.generateTFtrainDataFromNParraysCANN_expXUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses, generatePositiveSamples=False)					
					elif(XUANNnegativeSamplesAll):
						#implementation limitation (sample negative contains a selection of experiences from all classes, not just negative classes) - this simplification deemed valid under assumptions: calculations will be averaged over large negative batch and numberClasses >> 2
						sampleNegativeData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)
						sampleNegativeDataList = []
						sampleNegativeDataList.append(sampleNegativeData)
					elif(XUANNnegativeSamplesRandom):
						sampleNegativeDataList = ANNtf2_algorithmCANN.generateTFtrainDataFromNParraysCANN_expXUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses, generatePositiveSamples=True)					
					samplePositiveDataListIterators = []
					for samplePositiveData in samplePositiveDataList:
						samplePositiveDataListIterators.append(iter(samplePositiveData))
					sampleNegativeDataListIterators = []
					for sampleNegativeData in sampleNegativeDataList:
						sampleNegativeDataListIterators.append(iter(sampleNegativeData))
				else:
					trainData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)
					trainDataList = []
					trainDataList.append(trainData)		
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
			
			for batchIndex in range(int(trainingSteps)):
				(batchX, batchY) = trainDataListIterators[trainDataIndex].get_next()	#next(trainDataListIterators[trainDataIndex])
				
				batchYactual = batchY
				if(algorithm == "CANN"):
					if(algorithmCANN == "CANN_expAUANN"):
						(exemplarsX, exemplarsY) = exemplarDataListIterators[trainDataIndex].get_next()
						batchYactual = ANNtf2_algorithmCANN_expAUANN.generateTFYActualfromYandExemplarYCANN_expAUANN(batchY, exemplarsY)
					if(algorithmCANN == "CANN_expXUANN"):
						(samplePositiveX, samplePositiveY) = samplePositiveDataListIterators[trainDataIndex].get_next()
						if(XUANNnegativeSamplesRandom):
							foundTrainDataIndexNegative = False
							while not foundTrainDataIndexNegative:
								trainDataIndexNegative = np.random.randint(0, datasetNumClasses)
								if(trainDataIndexNegative != trainDataIndex):
									foundTrainDataIndexNegative = True
							(sampleNegativeX, sampleNegativeY) = sampleNegativeDataListIterators[trainDataIndexNegative].get_next()
						else:
							(sampleNegativeX, sampleNegativeY) = sampleNegativeDataListIterators[trainDataIndex].get_next()
														
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
					elif(algorithm == "SANI"):
						#learning algorithm not yet implemented:
						if(batchSize > 1):
							pred = neuralNetworkPropagation(batchX)
						if(batchIndex % displayStep == 0):
							pred = neuralNetworkPropagation(batchX)
							acc = tf.reduce_mean(tf.dtypes.cast(pred, tf.float32))
							print("batchIndex: %i, accuracy: %f" % (batchIndex, acc))
					elif(algorithm == "CANN"):
						if(algorithmCANN == "CANN_expHUANN"):
							batchYoneHot = tf.one_hot(batchY, depth=datasetNumClasses)
							executeLearningCANN(batchX, batchYoneHot, networkIndex)
						elif(algorithmCANN == "CANN_expSUANN"):
							executeLearningCANN(batchX, batchY, networkIndex)
						elif(algorithmCANN == "CANN_expAUANN"):
							#learning algorithm embedded in forward propagation of new class x experience following forward propagation of existing class x experience
							executeLearningCANN_expAUANN(batchX, batchY, exemplarsX, exemplarsY, currentClassTarget, networkIndex)
						elif(algorithmCANN == "CANN_expCUANN"):
							executeLearningCANN(batchX, batchY, networkIndex)	#currentClassTarget
						elif(algorithmCANN == "CANN_expXUANN"):
							executeLearningCANN_expXUANN(batchX, batchY, samplePositiveX, samplePositiveY, sampleNegativeX, sampleNegativeY, networkIndex)
						elif(algorithmCANN == "CANN_expMUANN"):
							executeLearningCANN(batchX, batchY, networkIndex)
						elif(algorithmCANN == "CANN_expRUANN"):
							executeLearningCANN(batchX, batchY, networkIndex)
						if(batchIndex % displayStep == 0):
							pred = neuralNetworkPropagation(batchX, networkIndex)
							loss = crossEntropy(pred, batchYactual, datasetNumClasses, costCrossEntropyWithLogits)
							acc = calculateAccuracy(pred, batchYactual)
							print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
							predNetworkAverage = predNetworkAverage + pred
					elif(algorithm == "FBANN"):
						executeOptimisation(batchX, batchY, networkIndex)
						if(batchIndex % displayStep == 0):
							pred = neuralNetworkPropagation(batchX, networkIndex)
							#print("pred = ", pred)
							loss = crossEntropy(pred, batchY, datasetNumClasses, costCrossEntropyWithLogits)
							acc = calculateAccuracy(pred, batchY)
							print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
							predNetworkAverage = predNetworkAverage + pred
				
				if(algorithm == "CANN"):
					if(algorithmCANN == "CANN_expAUANN"):
						#batchYactual = ANNtf2_algorithmCANN_expAUANN.generateTFYActualfromYandExemplarYCANN_expAUANN(batchY, exemplarsY)
						currentClassTarget = currentClassTarget+1
						if(currentClassTarget == datasetNumClassTargets):
							currentClassTarget = 0
						trainDataIndex = currentClassTarget
					elif(algorithmCANN == "CANN_expXUANN"):
						currentClassTarget = currentClassTarget+1
						if(currentClassTarget == datasetNumClassTargets):
							currentClassTarget = 0
						trainDataIndex = currentClassTarget
							
				if(batchIndex % displayStep == 0):
					if(trainMultipleNetworks):
						predNetworkAverage = predNetworkAverage / numberOfNetworks
						loss = crossEntropy(predNetworkAverage, batchYactual, datasetNumClasses, costCrossEntropyWithLogits)
						acc = calculateAccuracy(predNetworkAverage, batchYactual)
						print("batchIndex: %i, loss: %f, accuracy: %f" % (batchIndex, loss, acc))	


			predNetworkAverageAll = tf.Variable(tf.zeros([test_y.shape[0], datasetNumClasses]))
			for networkIndex in range(1, numberOfNetworks+1):
				if(algorithm == "ANN"):
					pred = neuralNetworkPropagation(test_x, networkIndex)	#test_x batch may be too large to propagate simultaneously and require subdivision
					print("Test Accuracy: networkIndex: %i, %f" % (networkIndex, calculateAccuracy(pred, test_y)))
					predNetworkAverageAll = predNetworkAverageAll + pred
				elif(algorithm == "SANI"):
					#learning algorithm not yet implemented:
					pythonDummy = 1
				elif(algorithm == "CANN"):
					pred = neuralNetworkPropagation(test_x, networkIndex)
					print("Test Accuracy: networkIndex: %i, %f" % (networkIndex, calculateAccuracy(pred, test_y)))
					predNetworkAverageAll = predNetworkAverageAll + pred
				elif(algorithm == "FBANN"):
					pred = neuralNetworkPropagation(test_x, networkIndex)
					print("Test Accuracy: networkIndex: %i, %f" % (networkIndex, calculateAccuracy(pred, test_y)))
					predNetworkAverageAll = predNetworkAverageAll + pred				

			if(trainMultipleNetworks):
					predNetworkAverageAll = predNetworkAverageAll / numberOfNetworks
					#print("predNetworkAverageAll", predNetworkAverageAll)
					acc = calculateAccuracy(predNetworkAverageAll, test_y)
					print("Test Accuracy: %f" % (acc))
