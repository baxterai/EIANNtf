"""ANNtf2.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
Python 3 and Tensorflow 2.1+ 

conda create -n anntf2 python=3.7
source activate anntf2
conda install -c tensorflow tensorflow=2.3
conda install scikit-learn (ANNtf2_algorithmLIANN_math:SVD/PCA only)
	
# Usage:
python3 ANNtf2.py

# Description:
ANNtf - train an experimental artificial neural network (ANN/LREANN/FBANN/EIANN/BAANN/LIANN/AEANN)

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

#select algorithm:
algorithm = "ANN"	#standard artificial neural network (backprop)
#algorithm = "LREANN"	#learning rule experiment artificial neural network
#algorithm = "FBANN"	#feedback artificial neural network (reverse connectivity)	#incomplete
#algorithm = "EIANN"	#excitatory/inhibitory artificial neural network	#incomplete+non-convergent
#algorithm = "BAANN"	#breakaway artificial neural network
#algorithm = "LIANN"	#local inhibition artificial neural network	#incomplete+non-convergent
#algorithm = "AEANN"	#autoencoder generated artificial neural network

suppressGradientDoNotExistForVariablesWarnings = True

costCrossEntropyWithLogits = False
if(algorithm == "ANN"):
	import ANNtf2_algorithmANN as ANNtf2_algorithm
elif(algorithm == "LREANN"):
	#select algorithmLREANN:
	#algorithmLREANN = "LREANN_expHUANN"	#incomplete+non-convergent
	#algorithmLREANN = "LREANN_expSUANN"	
	#algorithmLREANN = "LREANN_expAUANN"	#incomplete+non-convergent
	#algorithmLREANN = "LREANN_expCUANN"	#incomplete+non-convergent
	#algorithmLREANN = "LREANN_expXUANN"	#incomplete
	#algorithmLREANN = "LREANN_expMUANN"	#incomplete+non-convergent
	algorithmLREANN = "LREANN_expRUANN"
	if(algorithmLREANN == "LREANN_expHUANN"):
		import ANNtf2_algorithmLREANN_expHUANN as ANNtf2_algorithm
	elif(algorithmLREANN == "LREANN_expSUANN"):
		import ANNtf2_algorithmLREANN_expSUANN as ANNtf2_algorithm
	elif(algorithmLREANN == "LREANN_expAUANN"):
		import ANNtf2_algorithmLREANN_expAUANN as ANNtf2_algorithm
	elif(algorithmLREANN == "LREANN_expCUANN"):
		import ANNtf2_algorithmLREANN_expCUANN as ANNtf2_algorithm
	elif(algorithmLREANN == "LREANN_expXUANN"):
		XUANNnegativeSamplesComplement = False	#default: True
		XUANNnegativeSamplesAll = False	#default: False #orig implementation
		XUANNnegativeSamplesRandom = True	#default: False 
		import ANNtf2_algorithmLREANN_expXUANN as ANNtf2_algorithm
	elif(algorithmLREANN == "LREANN_expMUANN"):
		import ANNtf2_algorithmLREANN_expMUANN as ANNtf2_algorithm		
	elif(algorithmLREANN == "LREANN_expRUANN"):
		import ANNtf2_algorithmLREANN_expRUANN as ANNtf2_algorithm
elif(algorithm == "FBANN"):
	import ANNtf2_algorithmFBANN as ANNtf2_algorithm
elif(algorithm == "EIANN"):
	import ANNtf2_algorithmEIANN as ANNtf2_algorithm
elif(algorithm == "BAANN"):
	import ANNtf2_algorithmBAANN as ANNtf2_algorithm
elif(algorithm == "LIANN"):
	import ANNtf2_algorithmLIANN as ANNtf2_algorithm
elif(algorithm == "AEANN"):
	import ANNtf2_algorithmAEANN as ANNtf2_algorithm
	
						
#learningRate, trainingSteps, batchSize, displayStep, numEpochs = -1

#performance enhancements for development environment only: 
debugUseSmallPOStagSequenceDataset = True	#def:False	#switch increases performance during development	#eg data-POStagSentence-smallBackup
useSmallSentenceLengths = True	#def:False	#switch increases performance during development	#eg data-simple-POStagSentence-smallBackup
trainMultipleFiles = False	#def:True	#switch increases performance during development	#eg data-POStagSentence
trainMultipleNetworks = False	#trial improve classification accuracy by averaging over multiple independently trained networks (test)
numberOfNetworks = 1
if(algorithm == "ANN"):
	#trainMultipleNetworks = True	#optional
	if(trainMultipleNetworks):
		numberOfNetworks = 5
	else:
		numberOfNetworks = 1

if(trainMultipleFiles):
	randomiseFileIndexParse = True
	fileIndexFirst = 0
	if(useSmallSentenceLengths):
		fileIndexLast = 11
	else:
		fileIndexLast = 1202	#defined by wiki database extraction size
else:
	randomiseFileIndexParse = False
				
#loadDatasetType3 parameters:
#if generatePOSunambiguousInput=True, generate POS unambiguous permutations for every POS ambiguous data example/experience
#if onlyAddPOSunambiguousInputToTrain=True, do not train network with ambiguous POS possibilities
#if generatePOSunambiguousInput=False and onlyAddPOSunambiguousInputToTrain=False, requires simultaneous propagation of different (ambiguous) POS possibilities

if(algorithm == "ANN"):
	dataset = "SmallDataset"
	#trainMultipleNetworks = True	#default: False
	#numberOfNetworks = 3	#default: 1	
elif(algorithm == "LREANN"):
	dataset = "SmallDataset"
	#trainMultipleNetworks = True	#default: False
	#numberOfNetworks = 5	#default: 1
	trainHebbianBackprop = False	#default: False
elif(algorithm == "FBANN"):
	dataset = "SmallDataset"
	#trainMultipleNetworks = True	#default: False
	#numberOfNetworks = 5	#default: 1
elif(algorithm == "EIANN"):
	dataset = "SmallDataset"
	#trainMultipleNetworks = True	#default: False
	#numberOfNetworks = 3	#default: 1		
elif(algorithm == "BAANN"):
	dataset = "SmallDataset"
	#trainMultipleNetworks = True	#default: False
	#numberOfNetworks = 3	#default: 1		
elif(algorithm == "LIANN"):
	dataset = "SmallDataset"
	#trainMultipleNetworks = True	#default: False
	#numberOfNetworks = 3	#default: 1		
elif(algorithm == "AEANN"):
	dataset = "SmallDataset"
	#trainMultipleNetworks = True	#default: False
	#numberOfNetworks = 3	#default: 1				
		
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
	dataset2FileName = smallDatasetDefinitions[smallDatasetIndex][smallDatasetDefinitionsHeader['fileName']]
	datasetClassColumnFirst = smallDatasetDefinitions[smallDatasetIndex][smallDatasetDefinitionsHeader['classColumnFirst']]
	print("dataset2FileName = ", dataset2FileName)
	print("datasetClassColumnFirst = ", datasetClassColumnFirst)
			
if(debugUseSmallPOStagSequenceDataset):
	dataset1FileNameXstart = "Xdataset1PartSmall"
	dataset1FileNameYstart = "Ydataset1PartSmall"
	dataset3FileNameXstart = "Xdataset3PartSmall"
	dataset4FileNameStart = "Xdataset4PartSmall"
else:
	dataset1FileNameXstart = "Xdataset1Part"
	dataset1FileNameYstart = "Ydataset1Part"
	dataset3FileNameXstart = "Xdataset3Part"
	dataset4FileNameStart = "Xdataset4Part"
datasetFileNameXend = ".dat"
datasetFileNameYend = ".dat"
datasetFileNameStart = "datasetPart"
datasetFileNameEnd = ".dat"
xmlDatasetFileNameEnd = ".xml"


def defineTrainingParameters(dataset, numberOfFeaturesPerWord=None, paddingTagIndex=None):
	return ANNtf2_algorithm.defineTrainingParameters(dataset)

def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, useSmallSentenceLengths=None, numberOfFeaturesPerWord=None):
	return ANNtf2_algorithm.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks)	

def defineNeuralNetworkParameters():
	return ANNtf2_algorithm.defineNeuralNetworkParameters()

#define default forward prop function for test (identical to below);
def neuralNetworkPropagationTest(test_x, networkIndex=1):
	return ANNtf2_algorithm.neuralNetworkPropagation(test_x, networkIndex)

#define default forward prop function for backprop weights optimisation;
def neuralNetworkPropagation(x, networkIndex=1, l=None):
	return ANNtf2_algorithm.neuralNetworkPropagation(x, networkIndex)


#define specific learning algorithms (non-backprop);
#algorithm LREANN:
def executeLearningLREANN(x, y, networkIndex=1):
	if(algorithmLREANN == "LREANN_expHUANN"):
		#learning algorithm embedded in forward propagation
		if(trainHebbianBackprop):
			pred = ANNtf2_algorithm.neuralNetworkPropagationLREANN_expHUANNtrain(x, y, networkIndex, trainHebbianBackprop=True, trainHebbianLastLayerSupervision=True)
		else:
			pred = ANNtf2_algorithm.neuralNetworkPropagationLREANN_expHUANNtrain(x, y, networkIndex, trainHebbianForwardprop=True, trainHebbianLastLayerSupervision=True)
	elif(algorithmLREANN == "LREANN_expSUANN"):
		#learning algorithm embedded in multiple iterations of forward propagation
		pred = ANNtf2_algorithm.neuralNetworkPropagationLREANN_expSUANNtrain(x, y, networkIndex)
	elif(algorithmLREANN == "LREANN_expCUANN"):
		#learning algorithm embedded in forward propagation of new class x experience following forward propagation of existing class x experience
		pred = ANNtf2_algorithm.neuralNetworkPropagationLREANN_expCUANNtrain(x, y, networkIndex)
	elif(algorithmLREANN == "LREANN_expMUANN"):
		#learning algorithm embedded in multiple forward propagation and synaptic delta calculations
		pred = ANNtf2_algorithm.neuralNetworkPropagationLREANN_expMUANNtrain(x, y, networkIndex)
	elif(algorithmLREANN == "LREANN_expRUANN"):
		#learning algorithm: in reverse order, stochastically establishing Aideal of each layer (by temporarily biasing firing rate of neurons) to better achieve Aideal of higher layer (through multiple local/single layer forward propagations), then (simultaneous/parallel layer processing) stochastically adjusting weights to fine tune towards Aideal of their higher layers
		pred = ANNtf2_algorithm.neuralNetworkPropagationLREANN_expRUANNtrain(x, y, networkIndex)
def executeLearningLREANN_expAUANN(x, y, exemplarsX, exemplarsY, currentClassTarget, networkIndex=1):
	#learning algorithm embedded in forward propagation of new class x experience following forward propagation of existing class x experience
	pred = ANNtf2_algorithm.neuralNetworkPropagationLREANN_expAUANNtrain(x, y, exemplarsX, exemplarsY, currentClassTarget, networkIndex)
def executeLearningLREANN_expXUANN(x, y, samplePositiveX, samplePositiveY, sampleNegativeX, sampleNegativeY, networkIndex):
	#learning algorithm: perform contrast training (diff of interclass experience with current experience, and diff of extraclass experience with current experience) at each layer of network
	pred = ANNtf2_algorithm.neuralNetworkPropagationLREANN_expXUANNtrain(x, y, samplePositiveX, samplePositiveY, sampleNegativeX, sampleNegativeY, networkIndex)

#algorithm !LREANN:
def trainBatch(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, costCrossEntropyWithLogits, display, l=None):
	
	if(algorithm == "ANN"):
		executeOptimisation(batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex)
	elif(algorithm == "LREANN"):
		print("trainBatch error: does not support algorithm == LREANN; use trainLRE instead")
	elif(algorithm == "FBANN"):
		executeOptimisation(batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex)
	elif(algorithm == "EIANN"):
		if(ANNtf2_algorithm.learningAlgorithmFinalLayerBackpropHebbian):
			#first learning algorithm: perform neuron independence training
			batchYoneHot = tf.one_hot(batchY, depth=datasetNumClasses)
			executeLearningEIANN(batchX, batchYoneHot, networkIndex)
			#second learning algorithm (final layer hebbian connections to output class targets):
		executeOptimisation(batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex)
	elif(algorithm == "LIANN"):
		#first learning algorithm: perform neuron independence training
		batchYoneHot = tf.one_hot(batchY, depth=datasetNumClasses)
		executeLearningLIANN(batchX, batchYoneHot, networkIndex)
		if(ANNtf2_algorithm.learningAlgorithmFinalLayerBackpropHebbian):
			#second learning algorithm (final layer hebbian connections to output class targets):
			executeOptimisation(batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex)	
	elif(algorithm == "AEANN"):
		#print("trainMultipleFiles error: does not support greedy training for AEANN")
		#for l in range(1, numberOfLayers+1):
		executeOptimisation(batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, l=l)

	pred = None
	if(display):
		loss, acc = calculatePropagationLoss(batchX, batchY, datasetNumClasses, numberOfLayers, costCrossEntropyWithLogits, networkIndex, l)
		if(l is not None):
			print("l: %i, networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (l, networkIndex, batchIndex, loss, acc))			
		else:
			print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))

	return pred
			
def executeLearningEIANN(x, y, networkIndex):
	#first learning algorithm: perform neuron independence training
	pred = ANNtf2_algorithm.neuralNetworkPropagationEIANNtrain(x, networkIndex)
def executeLearningLIANN(x, y, networkIndex):
	#first learning algorithm: perform neuron independence training
	pred = ANNtf2_algorithm.neuralNetworkPropagationLIANNtrain(x, networkIndex)
#def executeLearningAEANN(x, y, networkIndex):
#	#first learning algorithm: perform neuron independence training
#	pred = ANNtf2_algorithm.neuralNetworkPropagationAEANNtrain(x, networkIndex)

		
def executeOptimisation(x, y, datasetNumClasses, numberOfLayers, optimizer, networkIndex=1, l=None):
	with tf.GradientTape() as gt:
		loss, acc = calculatePropagationLoss(x, y, datasetNumClasses, numberOfLayers, costCrossEntropyWithLogits, networkIndex, l)
		
	if(algorithm == "ANN"):
		Wlist = []
		Blist = []
		for l1 in range(1, numberOfLayers+1):
			if(ANNtf2_algorithm.supportSkipLayers):
				for l2 in range(0, l1):
					if(l2 < l1):
						Wlist.append(ANNtf2_algorithm.W[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "W")])
				Blist.append(ANNtf2_algorithm.B[generateParameterNameNetwork(networkIndex, l1, "B")])			
			else:
				if(ANNtf2_algorithm.debugOnlyTrainFinalLayer):
					if(l1 == numberOfLayers):
						Wlist.append(ANNtf2_algorithm.W[generateParameterNameNetwork(networkIndex, l1, "W")])
						Blist.append(ANNtf2_algorithm.B[generateParameterNameNetwork(networkIndex, l1, "B")])				
				else:	
					Wlist.append(ANNtf2_algorithm.W[generateParameterNameNetwork(networkIndex, l1, "W")])
					Blist.append(ANNtf2_algorithm.B[generateParameterNameNetwork(networkIndex, l1, "B")])
		trainableVariables = Wlist + Blist
		WlistLength = len(Wlist)
		BlistLength = len(Blist)
	elif(algorithm == "FBANN"):
		Wflist = []
		Wblist = []
		Blist = []
		for l1 in range(1, ANNtf2_algorithm.highestLayer+1):
			if(ANNtf2_algorithm.supportSkipLayers):
				for l2 in range(0, l1):
					if(l2 < l1):
						Wflist.append(ANNtf2_algorithm.Wf[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wf")])
				if(ANNtf2_algorithm.feedbackConnections):
					if((l1 <= ANNtf2_algorithm.highestLayerWithIncomingBackwardsConnections) and (l1 >= ANNtf2_algorithm.lowestLayerWithIncomingBackwardsConnections)):
						for l2 in range(l1+1, ANNtf2_algorithm.highestLayer+1):
							if(l2 > l1):
								Wblist.append(ANNtf2_algorithm.Wb[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "Wb")])
			else:
				Wflist.append(ANNtf2_algorithm.Wf[generateParameterNameNetwork(networkIndex, l1, "Wf")])
				if(ANNtf2_algorithm.feedbackConnections):
					if((l1 <= ANNtf2_algorithm.highestLayerWithIncomingBackwardsConnections) and (l1 >= ANNtf2_algorithm.lowestLayerWithIncomingBackwardsConnections)):
						Wblist.append(ANNtf2_algorithm.Wb[generateParameterNameNetwork(networkIndex, l1, "Wb")])
								
			Blist.append(ANNtf2_algorithm.B[generateParameterNameNetwork(networkIndex, l1, "B")])
			
		if(ANNtf2_algorithm.feedbackConnections):
			trainableVariables = Wflist + Wblist + Blist
		else:
			trainableVariables = Wflist + Blist
	elif(algorithm == "LREANN"):
		print("executeOptimisation error: algorithm LREANN not supported, use executeLearningLREANN() instead")
		exit()
	elif(algorithm == "EIANN"):
		Wlist = []
		Blist = []
		for l in range(1, numberOfLayers+1):
			if(ANNtf2_algorithm.learningAlgorithmFinalLayerBackpropHebbian):
				if(l == numberOfLayers):
					Wlist.append(ANNtf2_algorithm.W[generateParameterNameNetwork(networkIndex, l, "W")])
					Blist.append(ANNtf2_algorithm.B[generateParameterNameNetwork(networkIndex, l, "B")])				
			else:	
				Wlist.append(ANNtf2_algorithm.W[generateParameterNameNetwork(networkIndex, l, "W")])
				Blist.append(ANNtf2_algorithm.B[generateParameterNameNetwork(networkIndex, l, "B")])
		trainableVariables = Wlist + Blist
		WlistLength = len(Wlist)
		BlistLength = len(Blist)
	elif(algorithm == "LIANN"):
		#second learning algorithm (final layer hebbian connections to output class targets):
		Wlist = []
		Blist = []
		for l in range(1, numberOfLayers+1):
			if(ANNtf2_algorithm.learningAlgorithmFinalLayerBackpropHebbian):
				if(l == numberOfLayers):
					Wlist.append(ANNtf2_algorithm.W[generateParameterNameNetwork(networkIndex, l, "W")])
					Blist.append(ANNtf2_algorithm.B[generateParameterNameNetwork(networkIndex, l, "B")])				
			else:	
				Wlist.append(ANNtf2_algorithm.W[generateParameterNameNetwork(networkIndex, l, "W")])
				Blist.append(ANNtf2_algorithm.B[generateParameterNameNetwork(networkIndex, l, "B")])
		trainableVariables = Wlist + Blist
		WlistLength = len(Wlist)
		BlistLength = len(Blist)
	elif(algorithm == "AEANN"):
		#train specific layer weights;
		Wlist = []
		Blist = []
		if(l == numberOfLayers):
			Wlist.append(ANNtf2_algorithm.Wf[generateParameterNameNetwork(networkIndex, l, "Wf")])
			Blist.append(ANNtf2_algorithm.B[generateParameterNameNetwork(networkIndex, l, "B")])		
		else:
			Wlist.append(ANNtf2_algorithm.Wf[generateParameterNameNetwork(networkIndex, l, "Wf")])
			Wlist.append(ANNtf2_algorithm.Wb[generateParameterNameNetwork(networkIndex, l, "Wb")])
			Blist.append(ANNtf2_algorithm.B[generateParameterNameNetwork(networkIndex, l, "B")])
		trainableVariables = Wlist + Blist
		WlistLength = len(Wlist)
		BlistLength = len(Blist)
			
	gradients = gt.gradient(loss, trainableVariables)
						
	if(suppressGradientDoNotExistForVariablesWarnings):
		optimizer.apply_gradients([
    		(grad, var) 
    		for (grad, var) in zip(gradients, trainableVariables) 
    		if grad is not None
			])
	else:
		optimizer.apply_gradients(zip(gradients, trainableVariables))

	if(algorithm == "EIANN"):
		#set all W/B parameters to zero if their updated values violate the E/I neuron type condition
		for l in range(1, numberOfLayers+1):

			neuronEIlayerPrevious = ANNtf2_algorithm.neuronEI[generateParameterNameNetwork(networkIndex, l-1, "neuronEI")]
			neuronEIlayerPreviousTiled = tileDimension(neuronEIlayerPrevious, 1, ANNtf2_algorithm.n_h[l], True)
			neuronEI = ANNtf2_algorithm.neuronEI[generateParameterNameNetwork(networkIndex, l, "neuronEI")]

			Wlayer = ANNtf2_algorithm.W[generateParameterNameNetwork(networkIndex, l, "W")]
			WlayerSign = tf.sign(Wlayer)
			WlayerSignBool = convertSignOutputToBool(WlayerSign)
			Blayer = ANNtf2_algorithm.B[generateParameterNameNetwork(networkIndex, l, "B")]
			BlayerSign = tf.sign(Blayer)
			BlayerSignBool = convertSignOutputToBool(BlayerSign)
			
			WlayerSignCheck = tf.equal(WlayerSignBool, neuronEIlayerPreviousTiled)
			BlayerSignCheck = tf.equal(BlayerSignBool, neuronEI)
			
			#ignore 0.0 values in W/B arrays:
			WlayerSignCheck = tf.logical_or(WlayerSignCheck, tf.equal(WlayerSign, 0.0))
			BlayerSignCheck = tf.logical_or(BlayerSignCheck, tf.equal(BlayerSign, 0.0))
	
			WlayerCorrected = tf.where(WlayerSignCheck, Wlayer, 0.0)
			BlayerCorrected = tf.where(BlayerSignCheck, Blayer, 0.0)
			#print("WlayerCorrected = ", WlayerCorrected)	   
			#print("BlayerCorrected = ", BlayerCorrected)
						
			ANNtf2_algorithm.W[generateParameterNameNetwork(networkIndex, l, "W")] = WlayerCorrected
			if(l < numberOfLayers):
				ANNtf2_algorithm.B[generateParameterNameNetwork(networkIndex, l, "B")] = BlayerCorrected

		#excitatory/inhibitory weight verification (in accordance with neuron types):	
		for l in range(1, numberOfLayers+1):

			neuronEIlayerPrevious = ANNtf2_algorithm.neuronEI[generateParameterNameNetwork(networkIndex, l-1, "neuronEI")]
			neuronEIlayerPreviousTiled = tileDimension(neuronEIlayerPrevious, 1, ANNtf2_algorithm.n_h[l], True)
			neuronEI = ANNtf2_algorithm.neuronEI[generateParameterNameNetwork(networkIndex, l, "neuronEI")]

			Wlayer = ANNtf2_algorithm.W[generateParameterNameNetwork(networkIndex, l, "W")]
			WlayerSign = tf.sign(Wlayer)
			WlayerSignBool = convertSignOutputToBool(WlayerSign)
			Blayer = ANNtf2_algorithm.B[generateParameterNameNetwork(networkIndex, l, "B")]
			BlayerSign = tf.sign(Blayer)
			BlayerSignBool = convertSignOutputToBool(BlayerSign)
			
			WlayerSignCheck = tf.equal(WlayerSignBool, neuronEIlayerPreviousTiled)
			BlayerSignCheck = tf.equal(BlayerSignBool, neuronEI)
			
			#ignore 0.0 values in W/B arrays:
			WlayerSignCheck = tf.logical_or(WlayerSignCheck, tf.equal(WlayerSign, 0.0))
			BlayerSignCheck = tf.logical_or(BlayerSignCheck, tf.equal(BlayerSign, 0.0))

			WlayerSignCheck = tf.math.reduce_all(WlayerSignCheck).numpy()
			BlayerSignCheck = tf.math.reduce_all(WlayerSignCheck).numpy()
			
			#print("WlayerSignCheck = ", WlayerSignCheck)	   
			#print("BlayerSignCheck = ", BlayerSignCheck)
			#print("Wlayer = ", Wlayer)	   
			#print("Blayer = ", Blayer)
					
			if(not WlayerSignCheck):
			   print("!WlayerSignCheck, l = ", l)
			   print("neuronEIlayerPrevious = ", neuronEIlayerPrevious)
			   print("Wlayer = ", Wlayer)
			if(not BlayerSignCheck):
			   print("!BlayerSignCheck, l = ", l)
			   print("neuronEI = ", neuronEI)
			   print("Blayer = ", Blayer)
						

def calculatePropagationLoss(x, y, datasetNumClasses, numberOfLayers, costCrossEntropyWithLogits, networkIndex=1, l=None):
	acc = 0	#only valid for softmax class targets 
	if(algorithm == "AEANN"):
		if(l == numberOfLayers):
			pred = ANNtf2_algorithm.neuralNetworkPropagationAEANNfinalLayer(x, networkIndex)
			target = y 
			loss = calculateLossCrossEntropy(pred, target, datasetNumClasses, costCrossEntropyWithLogits)
			acc = calculateAccuracy(pred, target)	#only valid for softmax class targets 
			#print("target = ", target)
			#print("pred = ", pred)
			#print("2 loss = ", loss)
		else:
			pred = ANNtf2_algorithm.neuralNetworkPropagationAEANNautoencoderLayer(x, l, networkIndex)
			target = ANNtf2_algorithm.neuralNetworkPropagationAEANNtestLayer(x, l-1, autoencoder=False, networkIndex=networkIndex)
			loss = calculateLossMeanSquaredError(pred, target)
			#print("target = ", target)
			#print("pred = ", pred)
			#print("1 loss = ", loss)
	else:
		pred = neuralNetworkPropagation(x, networkIndex, l)
		target = y
		loss = calculateLossCrossEntropy(pred, target, datasetNumClasses, costCrossEntropyWithLogits)	
		acc = calculateAccuracy(pred, target)	#only valid for softmax class targets 

	return loss, acc
	
	
def loadDataset(fileIndex):

	global numberOfFeaturesPerWord
	global paddingTagIndex
	
	datasetNumFeatures = 0
	datasetNumClasses = 0
	
	fileIndexStr = str(fileIndex).zfill(4)
	if(dataset == "POStagSequence"):
		datasetType1FileNameX = dataset1FileNameXstart + fileIndexStr + datasetFileNameXend
		datasetType1FileNameY = dataset1FileNameYstart + fileIndexStr + datasetFileNameYend
	elif(dataset == "POStagSentence"):
		datasetType3FileNameX = dataset3FileNameXstart + fileIndexStr + datasetFileNameXend		
	elif(dataset == "SmallDataset"):
		if(trainMultipleFiles):
			datasetType2FileName = dataset2FileNameStart + fileIndexStr + datasetFileNameEnd
		else:
			datasetType2FileName = dataset2FileName
	elif(dataset == "wikiXmlDataset"):
		datasetType4FileName = dataset4FileNameStart + fileIndexStr + xmlDatasetFileNameEnd
			
	numberOfLayers = 0
	if(dataset == "POStagSequence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType1(datasetType1FileNameX, datasetType1FileNameY)
	elif(dataset == "POStagSentence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType3(datasetType3FileNameX, generatePOSunambiguousInput, onlyAddPOSunambiguousInputToTrain, useSmallSentenceLengths)
	elif(dataset == "SmallDataset"):
		datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType2(datasetType2FileName, datasetClassColumnFirst)
		numberOfFeaturesPerWord = None
		paddingTagIndex = None
	elif(dataset == "wikiXmlDataset"):
		articles = ANNtf2_loadDataset.loadDatasetType4(datasetType4FileName, AEANNsequentialInputTypesMaxLength, useSmallSentenceLengths, AEANNsequentialInputTypeMinWordVectors)

	if(dataset == "wikiXmlDataset"):
		return articles
	else:
		return numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp
		


#trainMinimal is minimal template code extracted from train based on trainMultipleNetworks=False, trainMultipleFiles=False, greedy=False;
def trainMinimal():
	
	networkIndex = 1	#assert numberOfNetworks = 1
	fileIndexTemp = 0	#assert trainMultipleFiles = False
	
	#generate network parameters based on dataset properties:
	numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = loadDataset(fileIndexTemp)

	#Model constants
	num_input_neurons = datasetNumFeatures  #train_x.shape[1]
	num_output_neurons = datasetNumClasses

	learningRate, trainingSteps, batchSize, displayStep, numEpochs = defineTrainingParameters(dataset, numberOfFeaturesPerWord, paddingTagIndex)
	numberOfLayers = defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, useSmallSentenceLengths, numberOfFeaturesPerWord)
	defineNeuralNetworkParameters()
														
	#stochastic gradient descent optimizer
	optimizer = tf.optimizers.SGD(learningRate)
	
	for e in range(numEpochs):

		print("epoch e = ", e)
	
		fileIndex = 0
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = loadDataset(fileIndex)

		shuffleSize = datasetNumExamples	#heuristic: 10*batchSize
		trainDataIndex = 0

		trainData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)
		trainDataList = []
		trainDataList.append(trainData)
		trainDataListIterators = []
		for trainData in trainDataList:
			trainDataListIterators.append(iter(trainData))
		testBatchX, testBatchY = generateTFbatch(test_x, test_y, batchSize)
		
		for batchIndex in range(int(trainingSteps)):
			(batchX, batchY) = trainDataListIterators[trainDataIndex].get_next()	#next(trainDataListIterators[trainDataIndex])
			batchYactual = batchY
					
			display = False
			if(batchIndex % displayStep == 0):
				display = True	
			trainBatch(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, costCrossEntropyWithLogits, display)

		pred = neuralNetworkPropagationTest(testBatchX, networkIndex)
		print("Test Accuracy: %f" % (calculateAccuracy(pred, testBatchY)))

			
#this function can be used to extract a minimal template (does not support algorithm==LREANN);
def train(trainMultipleNetworks=False, trainMultipleFiles=False, greedy=False):
	
	networkIndex = 1	#assert numberOfNetworks = 1
	fileIndexTemp = 0	#assert trainMultipleFiles = False
	
	#generate network parameters based on dataset properties:
	numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = loadDataset(fileIndexTemp)

	#Model constants
	num_input_neurons = datasetNumFeatures  #train_x.shape[1]
	num_output_neurons = datasetNumClasses

	learningRate, trainingSteps, batchSize, displayStep, numEpochs = defineTrainingParameters(dataset, numberOfFeaturesPerWord, paddingTagIndex)
	numberOfLayers = defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, useSmallSentenceLengths, numberOfFeaturesPerWord)
	defineNeuralNetworkParameters()

	#configure optional parameters;
	if(trainMultipleNetworks):
		maxNetwork = numberOfNetworks
	else:
		maxNetwork = 1
	if(trainMultipleFiles):
		minFileIndex = fileIndexFirst
		maxFileIndex = fileIndexLast
	else:
		minFileIndex = 0
		maxFileIndex = 0
	if(greedy):
		maxLayer = numberOfLayers
	else:
		maxLayer = 1
														
	#stochastic gradient descent optimizer
	optimizer = tf.optimizers.SGD(learningRate)
	
	for e in range(numEpochs):

		print("epoch e = ", e)
	
		#fileIndex = 0
		#trainMultipleFiles code;
		if(randomiseFileIndexParse):
			fileIndexShuffledArray = generateRandomisedIndexArray(fileIndexFirst, fileIndexLast)
		for f in range(minFileIndex, maxFileIndex+1):
			if(randomiseFileIndexParse):
				fileIndex = fileIndexShuffledArray[f]
			else:
				fileIndex = f
				
			numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = loadDataset(fileIndex)

			shuffleSize = datasetNumExamples	#heuristic: 10*batchSize
			trainDataIndex = 0

			#greedy code;
			for l in range(1, maxLayer+1):
				print("l = ", l)
				trainData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)
				trainDataList = []
				trainDataList.append(trainData)
				trainDataListIterators = []
				for trainData in trainDataList:
					trainDataListIterators.append(iter(trainData))
				testBatchX, testBatchY = generateTFbatch(test_x, test_y, batchSize)
				#testBatchX, testBatchY = (test_x, test_y)

				for batchIndex in range(int(trainingSteps)):
					(batchX, batchY) = trainDataListIterators[trainDataIndex].get_next()	#next(trainDataListIterators[trainDataIndex])
					batchYactual = batchY
					
					#trainMultipleNetworks code;
					predNetworkAverage = tf.Variable(tf.zeros(datasetNumClasses))
					for networkIndex in range(1, maxNetwork+1):

						display = False
						#if(l == maxLayer):	#only print accuracy after training final layer
						if(batchIndex % displayStep == 0):
							display = True	
						pred = trainBatch(batchIndex, batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, costCrossEntropyWithLogits, display, l)
						if(pred is not None):
							predNetworkAverage = predNetworkAverage + pred
						
					#trainMultipleNetworks code;
					if(trainMultipleNetworks):
						if(display):
							predNetworkAverage = predNetworkAverage / numberOfNetworks
							loss = calculateLossCrossEntropy(predNetworkAverage, batchYactual, datasetNumClasses, costCrossEntropyWithLogits)
							acc = calculateAccuracy(predNetworkAverage, batchYactual)
							print("batchIndex: %i, loss: %f, accuracy: %f" % (batchIndex, loss, acc))	

				#trainMultipleNetworks code;
				if(trainMultipleNetworks):
					predNetworkAverageAll = tf.Variable(tf.zeros([testBatchY.shape[0], datasetNumClasses]))
					for networkIndex in range(1, numberOfNetworks+1):
						pred = neuralNetworkPropagationTest(testBatchX, networkIndex)
						print("Test Accuracy: networkIndex: %i, %f" % (networkIndex, calculateAccuracy(pred, testBatchY)))
						predNetworkAverageAll = predNetworkAverageAll + pred
					predNetworkAverageAll = predNetworkAverageAll / numberOfNetworks
					#print("predNetworkAverageAll", predNetworkAverageAll)
					acc = calculateAccuracy(predNetworkAverageAll, testBatchY)
					print("Test Accuracy: %f" % (acc))
				else:
					pred = neuralNetworkPropagationTest(testBatchX, networkIndex)
					if(greedy):
						print("Test Accuracy: l: %i, %f" % (l, calculateAccuracy(pred, testBatchY)))
					else:
						print("Test Accuracy: %f" % (calculateAccuracy(pred, testBatchY)))

							
def trainLRE():
																
	#generate network parameters based on dataset properties:

	networkIndex = 1
	
	fileIndexTemp = 0	
	numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = loadDataset(fileIndexTemp)

	#Model constants
	num_input_neurons = datasetNumFeatures  #train_x.shape[1]
	num_output_neurons = datasetNumClasses
	if(algorithm == "LREANN"):
		if(algorithmLREANN == "LREANN_expAUANN"):
			num_output_neurons = ANNtf2_algorithm.calculateOutputNeuronsLREANN_expAUANN(datasetNumClasses)

	learningRate, trainingSteps, batchSize, displayStep, numEpochs = defineTrainingParameters(dataset, numberOfFeaturesPerWord, paddingTagIndex)
	numberOfLayers = defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworks, useSmallSentenceLengths, numberOfFeaturesPerWord)
	defineNeuralNetworkParameters()
									
	noisySampleGeneration = False
	if(algorithm == "LREANN"):
		noisySampleGeneration, noisySampleGenerationNumSamples, noiseStandardDeviation = ANNtf2_algorithm.getNoisySampleGenerationNumSamples()
		if(noisySampleGeneration):
			batchXmultiples = tf.constant([noisySampleGenerationNumSamples, 1], tf.int32)
			batchYmultiples = tf.constant([noisySampleGenerationNumSamples], tf.int32)
			randomNormal = tf.initializers.RandomNormal()	#tf.initializers.RandomUniform(minval=-1, maxval=1)

	#stochastic gradient descent optimizer
	optimizer = tf.optimizers.SGD(learningRate)

	for e in range(numEpochs):

		print("epoch e = ", e)

		fileIndex = 0

		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_x, train_y, test_x, test_y = loadDataset(fileIndex)

		shuffleSize = datasetNumExamples	#heuristic: 10*batchSize

		#new iteration method (only required for algorithm == "LREANN_expAUANN/LREANN_expCUANN"):	
		datasetNumClassesActual = datasetNumClasses
		trainDataIndex = 0
		
		if(algorithm == "LREANN"):
			if(algorithmLREANN == "LREANN_expAUANN"):
				currentClassTarget = 0
				generateClassTargetExemplars = False
				if(e == 0):
					generateClassTargetExemplars = True
				networkIndex = 1 #note ANNtf2_algorithmLREANN_expAUANN doesn't currently support multiple networks
				trainDataList = ANNtf2_algorithm.generateTFtrainDataFromNParraysLREANN_expAUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses)
				exemplarDataList = ANNtf2_algorithm.generateTFexemplarDataFromNParraysLREANN_expAUANN(train_x, train_y, networkIndex, shuffleSize, batchSize, datasetNumClasses, generateClassTargetExemplars)
				test_y = ANNtf2_algorithm.generateYActualfromYLREANN_expAUANN(test_y, num_output_neurons)
				datasetNumClassTargets = datasetNumClasses
				datasetNumClasses = ANNtf2_algorithm.generateNumClassesActualLREANN_expAUANN(datasetNumClasses, num_output_neurons)
				exemplarDataListIterators = []
				for exemplarData in exemplarDataList:
					exemplarDataListIterators.append(iter(exemplarData))
			elif(algorithmLREANN == "LREANN_expCUANN"):
				trainDataList = ANNtf2_algorithm.generateTFtrainDataFromNParraysLREANN_expCUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses)
			elif(algorithmLREANN == "LREANN_expXUANN"):
				currentClassTarget = 0
				generateClassTargetExemplars = False
				if(e == 0):
					generateClassTargetExemplars = True
				trainDataList = ANNtf2_algorithm.generateTFtrainDataFromNParraysLREANN_expXUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses, generatePositiveSamples=True)
				datasetNumClassTargets = datasetNumClasses
				samplePositiveDataList = ANNtf2_algorithm.generateTFtrainDataFromNParraysLREANN_expXUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses, generatePositiveSamples=True)
				if(XUANNnegativeSamplesComplement):
					sampleNegativeDataList = ANNtf2_algorithm.generateTFtrainDataFromNParraysLREANN_expXUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses, generatePositiveSamples=False)					
				elif(XUANNnegativeSamplesAll):
					#implementation limitation (sample negative contains a selection of experiences from all classes, not just negative classes) - this simplification deemed valid under assumptions: calculations will be averaged over large negative batch and numberClasses >> 2
					sampleNegativeData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)	#CHECKTHIS
					sampleNegativeDataList = []
					sampleNegativeDataList.append(sampleNegativeData)
				elif(XUANNnegativeSamplesRandom):
					sampleNegativeDataList = ANNtf2_algorithm.generateTFtrainDataFromNParraysLREANN_expXUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses, generatePositiveSamples=True)					
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
			if(algorithm == "LREANN"):
				if(algorithmLREANN == "LREANN_expAUANN"):
					(exemplarsX, exemplarsY) = exemplarDataListIterators[trainDataIndex].get_next()
					batchYactual = ANNtf2_algorithm.generateTFYActualfromYandExemplarYLREANN_expAUANN(batchY, exemplarsY)
				if(algorithmLREANN == "LREANN_expXUANN"):
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

			#can move code to trainBatchLRE();
			if(algorithm == "LREANN"):
				if(algorithmLREANN == "LREANN_expHUANN"):
					batchYoneHot = tf.one_hot(batchY, depth=datasetNumClasses)
					executeLearningLREANN(batchX, batchYoneHot, networkIndex)
				elif(algorithmLREANN == "LREANN_expSUANN"):
					executeLearningLREANN(batchX, batchY, networkIndex)
				elif(algorithmLREANN == "LREANN_expAUANN"):
					#learning algorithm embedded in forward propagation of new class x experience following forward propagation of existing class x experience
					executeLearningLREANN_expAUANN(batchX, batchY, exemplarsX, exemplarsY, currentClassTarget, networkIndex)
				elif(algorithmLREANN == "LREANN_expCUANN"):
					executeLearningLREANN(batchX, batchY, networkIndex)	#currentClassTarget
				elif(algorithmLREANN == "LREANN_expXUANN"):
					executeLearningLREANN_expXUANN(batchX, batchY, samplePositiveX, samplePositiveY, sampleNegativeX, sampleNegativeY, networkIndex)
				elif(algorithmLREANN == "LREANN_expMUANN"):
					executeLearningLREANN(batchX, batchY, networkIndex)
				elif(algorithmLREANN == "LREANN_expRUANN"):
					executeLearningLREANN(batchX, batchY, networkIndex)
				if(batchIndex % displayStep == 0):
					pred = neuralNetworkPropagation(batchX, networkIndex)
					loss = calculateLossCrossEntropy(pred, batchYactual, datasetNumClasses, costCrossEntropyWithLogits)
					acc = calculateAccuracy(pred, batchYactual)
					print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
					predNetworkAverage = predNetworkAverage + pred

			if(algorithm == "LREANN"):
				if(algorithmLREANN == "LREANN_expAUANN"):
					#batchYactual = ANNtf2_algorithm.generateTFYActualfromYandExemplarYLREANN_expAUANN(batchY, exemplarsY)
					currentClassTarget = currentClassTarget+1
					if(currentClassTarget == datasetNumClassTargets):
						currentClassTarget = 0
					trainDataIndex = currentClassTarget
				elif(algorithmLREANN == "LREANN_expXUANN"):
					currentClassTarget = currentClassTarget+1
					if(currentClassTarget == datasetNumClassTargets):
						currentClassTarget = 0
					trainDataIndex = currentClassTarget

		pred = neuralNetworkPropagationTest(testBatchX, networkIndex)
		print("Test Accuracy: networkIndex: %i, %f" % (networkIndex, calculateAccuracy(pred, test_y)))

def generateRandomisedIndexArray(indexFirst, indexLast, arraySize=None):
	fileIndexArray = np.arange(indexFirst, indexLast+1, 1)
	#print("fileIndexArray = " + str(fileIndexArray))
	if(arraySize is None):
		np.random.shuffle(fileIndexArray)
		fileIndexRandomArray = fileIndexArray
	else:
		fileIndexRandomArray = random.sample(fileIndexArray.tolist(), arraySize)
	
	print("fileIndexRandomArray = " + str(fileIndexRandomArray))
	return fileIndexRandomArray
	
				
if __name__ == "__main__":
	if(algorithm == "ANN"):
		if(trainMultipleNetworks):
			train(trainMultipleNetworks=trainMultipleNetworks)
		else:
			trainMinimal()
	elif(algorithm == "LREANN"):
		trainLRE()
	elif(algorithm == "FBANN"):
		trainMinimal()		
	elif(algorithm == "EIANN"):
		trainMinimal()		
	elif(algorithm == "BAANN"):
		#current implemenation uses tf.keras (could be changed to tf);
		fileIndexTemp = 0
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = defineTrainingParameters(dataset)
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_x, train_y, test_x, test_y = loadDataset(fileIndexTemp)
		ANNtf2_algorithm.BAANNmain(train_x, train_y, test_x, test_y, datasetNumFeatures, datasetNumClasses, batchSize, trainingSteps, numEpochs)
	elif(algorithm == "LIANN"):
		trainMinimal()
	elif(algorithm == "AEANN"):
		train(greedy=True)
	else:
		print("main error: algorithm == unknown")
		
