"""ANNtf2.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2021 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
Python 3 and Tensorflow 2.1+ 

conda create -n anntf2 python=3.7
source activate anntf2
conda install -c tensorflow tensorflow=2.3
conda install scikit-learn (ANNtf2_algorithmLIANN_PCAsimulation only)
	
# Usage:
python3 ANNtf2.py

# Description:
ANNtf - train an experimental artificial neural network (ANN/SANI/LREANN/FBANN/EIANN/BAANN/LIANN/AEANN)

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
from ANNtf2_algorithmSANIglobalDefs import algorithmSANI

#select algorithm:
#algorithm = "ANN"	#standard artificial neural network (backprop)
#algorithm = "SANI"	#sequentially activated neuronal input artificial neural network	#incomplete+non-convergent
#algorithm = "LREANN"	#learning rule experiment artificial neural network
#algorithm = "FBANN"	#feedback artificial neural network (reverse connectivity)	#incomplete
#algorithm = "EIANN"	#excitatory/inhibitory artificial neural network	#incomplete+non-convergent
#algorithm = "BAANN"	#breakaway artificial neural network
#algorithm = "LIANN"	#local inhibition artificial neural network	#incomplete+non-convergent
algorithm = "AEANN"	#autoencoder artificial neural network	#incomplete+non-convergent?

suppressGradientDoNotExistForVariablesWarnings = True

costCrossEntropyWithLogits = False
if(algorithm == "ANN"):
	import ANNtf2_algorithmANN
elif(algorithm == "SANI"):
	#set algorithmSANI in ANNtf2_algorithmSANIoperations
	if(algorithmSANI == "sharedModulesHebbian"):
		import ANNtf2_algorithmSANIsharedModulesHebbian as ANNtf2_algorithmSANI
		#no cost function used
	elif(algorithmSANI == "sharedModulesBinary"):
		import ANNtf2_algorithmSANIsharedModulesBinary as ANNtf2_algorithmSANI
	elif(algorithmSANI == "sharedModules"):
		import ANNtf2_algorithmSANIsharedModules as ANNtf2_algorithmSANI
		costCrossEntropyWithLogits = True
	elif(algorithmSANI == "repeatedModules"):
		import ANNtf2_algorithmSANIrepeatedModules as ANNtf2_algorithmSANI
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
		import ANNtf2_algorithmLREANN_expHUANN as ANNtf2_algorithmLREANN
	elif(algorithmLREANN == "LREANN_expSUANN"):
		import ANNtf2_algorithmLREANN_expSUANN as ANNtf2_algorithmLREANN
	elif(algorithmLREANN == "LREANN_expAUANN"):
		import ANNtf2_algorithmLREANN_expAUANN as ANNtf2_algorithmLREANN
	elif(algorithmLREANN == "LREANN_expCUANN"):
		import ANNtf2_algorithmLREANN_expCUANN as ANNtf2_algorithmLREANN
	elif(algorithmLREANN == "LREANN_expXUANN"):
		XUANNnegativeSamplesComplement = False	#default: True
		XUANNnegativeSamplesAll = False	#default: False #orig implementation
		XUANNnegativeSamplesRandom = True	#default: False 
		import ANNtf2_algorithmLREANN_expXUANN as ANNtf2_algorithmLREANN
	elif(algorithmLREANN == "LREANN_expMUANN"):
		import ANNtf2_algorithmLREANN_expMUANN as ANNtf2_algorithmLREANN		
	elif(algorithmLREANN == "LREANN_expRUANN"):
		import ANNtf2_algorithmLREANN_expRUANN as ANNtf2_algorithmLREANN
elif(algorithm == "FBANN"):
	import ANNtf2_algorithmFBANN as ANNtf2_algorithmFBANN
elif(algorithm == "EIANN"):
	import ANNtf2_algorithmEIANN as ANNtf2_algorithmEIANN
elif(algorithm == "BAANN"):
	import ANNtf2_algorithmBAANN as ANNtf2_algorithmBAANN
elif(algorithm == "LIANN"):
	import ANNtf2_algorithmLIANN as ANNtf2_algorithmLIANN
elif(algorithm == "AEANN"):
	import ANNtf2_algorithmAEANN as ANNtf2_algorithmAEANN
	
						
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
	if(ANNtf2_algorithmSANI.algorithmSANI == "sharedModulesHebbian"):
		if(ANNtf2_algorithmSANI.SANIsharedModules):
			dataset = "POStagSentence"
			numberOfFeaturesPerWord = -1
			paddingTagIndex = -1
			generatePOSunambiguousInput = False
			onlyAddPOSunambiguousInputToTrain = False	#True
		else:
			print("!ANNtf2_algorithmSANI.SANIsharedModules")
			dataset = "POStagSequence"
	elif(ANNtf2_algorithmSANI.algorithmSANI == "sharedModulesBinary"):
		if(ANNtf2_algorithmSANI.SANIsharedModules):	#only implementation
			print("sharedModulesBinary")
			dataset = "POStagSentence"
			numberOfFeaturesPerWord = -1
			paddingTagIndex = -1	
			generatePOSunambiguousInput = False
			onlyAddPOSunambiguousInputToTrain = False	#True
	elif(ANNtf2_algorithmSANI.algorithmSANI == "sharedModules"):
		if(ANNtf2_algorithmSANI.SANIsharedModules):	#only implementation
			dataset = "POStagSentence"
			numberOfFeaturesPerWord = -1
			paddingTagIndex = -1
			if(ANNtf2_algorithmSANI.allowMultipleContributingSubinputsPerSequentialInput):
				generatePOSunambiguousInput = False
				onlyAddPOSunambiguousInputToTrain = False
			else:
				generatePOSunambiguousInput = False
				onlyAddPOSunambiguousInputToTrain = True
	elif(ANNtf2_algorithmSANI.algorithmSANI == "repeatedModules"):
		if(not ANNtf2_algorithmSANI.SANIsharedModules):	#only implementation
			dataset = "POStagSequence"
elif(algorithm == "ANN"):
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
else:
	dataset1FileNameXstart = "Xdataset1Part"
	dataset1FileNameYstart = "Ydataset1Part"
	dataset3FileNameXstart = "Xdataset3Part"
datasetFileNameXend = ".dat"
datasetFileNameYend = ".dat"
datasetFileNameStart = "datasetPart"
datasetFileNameEnd = ".dat"


	
	
def neuralNetworkPropagation(x, networkIndex=1, l=None):
	if(algorithm == "SANI"):
		pred = ANNtf2_algorithmSANI.neuralNetworkPropagationSANI(x)
	elif(algorithm == "ANN"):
		pred = ANNtf2_algorithmANN.neuralNetworkPropagationANN(x, networkIndex)
	elif(algorithm == "LREANN"):
		pred = ANNtf2_algorithmLREANN.neuralNetworkPropagationLREANN(x, networkIndex)
	elif(algorithm == "FBANN"):
		pred = ANNtf2_algorithmFBANN.neuralNetworkPropagationFBANNwrapper(x, networkIndex)
	elif(algorithm == "EIANN"):
		pred = ANNtf2_algorithmEIANN.neuralNetworkPropagationEIANN(x, networkIndex)
	elif(algorithm == "LIANN"):
		pred = ANNtf2_algorithmLIANN.neuralNetworkPropagationLIANNtest(x, networkIndex)
	return pred
	

#algorithm LREANN:
def executeLearningLREANN(x, y, networkIndex=1):
	if(algorithmLREANN == "LREANN_expHUANN"):
		#learning algorithm embedded in forward propagation
		if(trainHebbianBackprop):
			pred = ANNtf2_algorithmLREANN.neuralNetworkPropagationLREANN_expHUANNtrain(x, y, networkIndex, trainHebbianBackprop=True, trainHebbianLastLayerSupervision=True)
		else:
			pred = ANNtf2_algorithmLREANN.neuralNetworkPropagationLREANN_expHUANNtrain(x, y, networkIndex, trainHebbianForwardprop=True, trainHebbianLastLayerSupervision=True)
	elif(algorithmLREANN == "LREANN_expSUANN"):
		#learning algorithm embedded in multiple iterations of forward propagation
		pred = ANNtf2_algorithmLREANN.neuralNetworkPropagationLREANN_expSUANNtrain(x, y, networkIndex)
	elif(algorithmLREANN == "LREANN_expCUANN"):
		#learning algorithm embedded in forward propagation of new class x experience following forward propagation of existing class x experience
		pred = ANNtf2_algorithmLREANN.neuralNetworkPropagationLREANN_expCUANNtrain(x, y, networkIndex)
	elif(algorithmLREANN == "LREANN_expMUANN"):
		#learning algorithm embedded in multiple forward propagation and synaptic delta calculations
		pred = ANNtf2_algorithmLREANN.neuralNetworkPropagationLREANN_expMUANNtrain(x, y, networkIndex)
	elif(algorithmLREANN == "LREANN_expRUANN"):
		#learning algorithm: in reverse order, stochastically establishing Aideal of each layer (by temporarily biasing firing rate of neurons) to better achieve Aideal of higher layer (through multiple local/single layer forward propagations), then (simultaneous/parallel layer processing) stochastically adjusting weights to fine tune towards Aideal of their higher layers
		pred = ANNtf2_algorithmLREANN.neuralNetworkPropagationLREANN_expRUANNtrain(x, y, networkIndex)
def executeLearningLREANN_expAUANN(x, y, exemplarsX, exemplarsY, currentClassTarget, networkIndex=1):
	#learning algorithm embedded in forward propagation of new class x experience following forward propagation of existing class x experience
	pred = ANNtf2_algorithmLREANN.neuralNetworkPropagationLREANN_expAUANNtrain(x, y, exemplarsX, exemplarsY, currentClassTarget, networkIndex)
def executeLearningLREANN_expXUANN(x, y, samplePositiveX, samplePositiveY, sampleNegativeX, sampleNegativeY, networkIndex):
	#learning algorithm: perform contrast training (diff of interclass experience with current experience, and diff of extraclass experience with current experience) at each layer of network
	pred = ANNtf2_algorithmLREANN.neuralNetworkPropagationLREANN_expXUANNtrain(x, y, samplePositiveX, samplePositiveY, sampleNegativeX, sampleNegativeY, networkIndex)

def executeLearningEIANN(x, y, networkIndex):
	#first learning algorithm: perform neuron independence training
	pred = ANNtf2_algorithmEIANN.neuralNetworkPropagationEIANNtrain(x, networkIndex)
def executeLearningLIANN(x, y, networkIndex):
	#first learning algorithm: perform neuron independence training
	pred = ANNtf2_algorithmLIANN.neuralNetworkPropagationLIANNtrain(x, y, networkIndex)
#def executeLearningAEANN(x, y, networkIndex):
#	#first learning algorithm: perform neuron independence training
#	pred = ANNtf2_algorithmAEANN.neuralNetworkPropagationAEANNtrain(x, y, networkIndex)


			
def executeOptimisation(x, y, datasetNumClasses, numberOfLayers, optimizer, networkIndex=1, l=None):
	with tf.GradientTape() as gt:
		if(algorithm == "AEANN"):
			if(l == numberOfLayers):
				pred = ANNtf2_algorithmAEANN.neuralNetworkPropagationAEANNfinalLayer(x, networkIndex)
				target = y 
				loss = calculateLossCrossEntropy(pred, target, datasetNumClasses, costCrossEntropyWithLogits)
			else:
				pred = ANNtf2_algorithmAEANN.neuralNetworkPropagationAEANNautoencoderLayer(x, l, networkIndex)
				target = ANNtf2_algorithmAEANN.neuralNetworkPropagationAEANNtestLayer(x, l-1, networkIndex)
				loss = calculateLossMeanSquaredError(pred, target)
		else:
			pred = neuralNetworkPropagation(x, networkIndex, l)
			target = y
			loss = calculateLossCrossEntropy(pred, target, datasetNumClasses, costCrossEntropyWithLogits)
		
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
		WlistLength = len(Wlist)
		BlistLength = len(Blist)
	elif(algorithm == "SANI"):
		print("executeOptimisation error: algorithm SANI not supported, use neuralNetworkPropagation() instead")
		exit()
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
	elif(algorithm == "LREANN"):
		print("executeOptimisation error: algorithm LREANN not supported, use executeLearningLREANN() instead")
		exit()
	elif(algorithm == "EIANN"):
		Wlist = []
		Blist = []
		for l in range(1, numberOfLayers+1):
			if(ANNtf2_algorithmEIANN.learningAlgorithmFinalLayerBackpropHebbian):
				if(l == numberOfLayers):
					Wlist.append(ANNtf2_algorithmEIANN.W[generateParameterNameNetwork(networkIndex, l, "W")])
					Blist.append(ANNtf2_algorithmEIANN.B[generateParameterNameNetwork(networkIndex, l, "B")])				
			else:	
				Wlist.append(ANNtf2_algorithmEIANN.W[generateParameterNameNetwork(networkIndex, l, "W")])
				Blist.append(ANNtf2_algorithmEIANN.B[generateParameterNameNetwork(networkIndex, l, "B")])
		trainableVariables = Wlist + Blist
		WlistLength = len(Wlist)
		BlistLength = len(Blist)
	elif(algorithm == "LIANN"):
		#second learning algorithm (final layer hebbian connections to output class targets):
		Wlist = []
		Blist = []
		for l in range(1, numberOfLayers+1):
			if(ANNtf2_algorithmLIANN.learningAlgorithmFinalLayerBackpropHebbian):
				if(l == numberOfLayers):
					Wlist.append(ANNtf2_algorithmLIANN.W[generateParameterNameNetwork(networkIndex, l, "W")])
					Blist.append(ANNtf2_algorithmLIANN.B[generateParameterNameNetwork(networkIndex, l, "B")])				
			else:	
				Wlist.append(ANNtf2_algorithmLIANN.W[generateParameterNameNetwork(networkIndex, l, "W")])
				Blist.append(ANNtf2_algorithmLIANN.B[generateParameterNameNetwork(networkIndex, l, "B")])
		trainableVariables = Wlist + Blist
		WlistLength = len(Wlist)
		BlistLength = len(Blist)
	elif(algorithm == "AEANN"):
		#train specific layer weights;
		Wlist = []
		Blist = []
		if(l == numberOfLayers):
			Wlist.append(ANNtf2_algorithmAEANN.Wf[generateParameterNameNetwork(networkIndex, l, "Wf")])
			Blist.append(ANNtf2_algorithmAEANN.B[generateParameterNameNetwork(networkIndex, l, "B")])		
		else:
			Wlist.append(ANNtf2_algorithmAEANN.Wf[generateParameterNameNetwork(networkIndex, l, "Wf")])
			Wlist.append(ANNtf2_algorithmAEANN.Wb[generateParameterNameNetwork(networkIndex, l, "Wb")])
			Blist.append(ANNtf2_algorithmAEANN.B[generateParameterNameNetwork(networkIndex, l, "B")])
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

			neuronEIlayerPrevious = ANNtf2_algorithmEIANN.neuronEI[generateParameterNameNetwork(networkIndex, l-1, "neuronEI")]
			neuronEIlayerPreviousTiled = tileDimension(neuronEIlayerPrevious, 1, ANNtf2_algorithmEIANN.n_h[l], True)
			neuronEI = ANNtf2_algorithmEIANN.neuronEI[generateParameterNameNetwork(networkIndex, l, "neuronEI")]

			Wlayer = ANNtf2_algorithmEIANN.W[generateParameterNameNetwork(networkIndex, l, "W")]
			WlayerSign = tf.sign(Wlayer)
			WlayerSignBool = convertSignOutputToBool(WlayerSign)
			Blayer = ANNtf2_algorithmEIANN.B[generateParameterNameNetwork(networkIndex, l, "B")]
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
						
			ANNtf2_algorithmEIANN.W[generateParameterNameNetwork(networkIndex, l, "W")] = WlayerCorrected
			if(l < numberOfLayers):
				ANNtf2_algorithmEIANN.B[generateParameterNameNetwork(networkIndex, l, "B")] = BlayerCorrected

		#excitatory/inhibitory weight verification (in accordance with neuron types):	
		for l in range(1, numberOfLayers+1):

			neuronEIlayerPrevious = ANNtf2_algorithmEIANN.neuronEI[generateParameterNameNetwork(networkIndex, l-1, "neuronEI")]
			neuronEIlayerPreviousTiled = tileDimension(neuronEIlayerPrevious, 1, ANNtf2_algorithmEIANN.n_h[l], True)
			neuronEI = ANNtf2_algorithmEIANN.neuronEI[generateParameterNameNetwork(networkIndex, l, "neuronEI")]

			Wlayer = ANNtf2_algorithmEIANN.W[generateParameterNameNetwork(networkIndex, l, "W")]
			WlayerSign = tf.sign(Wlayer)
			WlayerSignBool = convertSignOutputToBool(WlayerSign)
			Blayer = ANNtf2_algorithmEIANN.B[generateParameterNameNetwork(networkIndex, l, "B")]
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
						

def loadDataset(fileIndex):

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

	numberOfLayers = 0
	if(dataset == "POStagSequence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType1(datasetType1FileNameX, datasetType1FileNameY)
	elif(dataset == "POStagSentence"):
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType3(datasetType3FileNameX, generatePOSunambiguousInput, onlyAddPOSunambiguousInputToTrain, useSmallSentenceLengths)
	elif(dataset == "SmallDataset"):
		datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp = ANNtf2_loadDataset.loadDatasetType2(datasetType2FileName, datasetClassColumnFirst)
		numberOfFeaturesPerWord = None
		paddingTagIndex = None
	
	return numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamples, train_xTemp, train_yTemp, test_xTemp, test_yTemp


def main():
																
	#generate network parameters based on dataset properties:

	#global learningRate, trainingSteps, batchSize, displayStep, numEpochs

	fileIndexTemp = 0	
	numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_xTemp, train_yTemp, test_xTemp, test_yTemp = loadDataset(fileIndexTemp)

	#Model constants
	num_input_neurons = datasetNumFeatures  #train_x.shape[1]
	num_output_neurons = datasetNumClasses
	if(algorithm == "LREANN"):
		if(algorithmLREANN == "LREANN_expAUANN"):
			num_output_neurons = ANNtf2_algorithmLREANN.calculateOutputNeuronsLREANN_expAUANN(datasetNumClasses)

	if(algorithm == "ANN"):
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = ANNtf2_algorithmANN.defineTrainingParametersANN(dataset)
		numberOfLayers = ANNtf2_algorithmANN.defineNetworkParametersANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworks)
		ANNtf2_algorithmANN.defineNeuralNetworkParametersANN()
	elif(algorithm == "SANI"):
		ANNtf2_algorithmSANI.defineTrainingParametersSANIsharedModules(numberOfFeaturesPerWord, paddingTagIndex)
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = ANNtf2_algorithmSANI.defineTrainingParametersSANIwrapper(dataset, trainMultipleFiles)
		ANNtf2_algorithmSANI.defineNetworkParametersSANIwrapper(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, useSmallSentenceLengths, numberOfFeaturesPerWord)
		ANNtf2_algorithmSANI.defineNeuralNetworkParametersSANI()
	elif(algorithm == "LREANN"):
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = ANNtf2_algorithmLREANN.defineTrainingParametersLREANN(dataset)
		numberOfLayers = ANNtf2_algorithmLREANN.defineNetworkParametersLREANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworks)
		ANNtf2_algorithmLREANN.defineNeuralNetworkParametersLREANN()
	elif(algorithm == "FBANN"):
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = ANNtf2_algorithmFBANN.defineTrainingParametersFBANN(dataset)
		numberOfLayers = ANNtf2_algorithmFBANN.defineNetworkParametersFBANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworks)
		ANNtf2_algorithmFBANN.defineNeuralNetworkParametersFBANN()
	elif(algorithm == "EIANN"):
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = ANNtf2_algorithmEIANN.defineTrainingParametersEIANN(dataset)
		numberOfLayers = ANNtf2_algorithmEIANN.defineNetworkParametersEIANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworks)
		ANNtf2_algorithmEIANN.defineNeuralNetworkParametersEIANN()
	elif(algorithm == "LIANN"):
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = ANNtf2_algorithmLIANN.defineTrainingParametersLIANN(dataset)
		numberOfLayers = ANNtf2_algorithmLIANN.defineNetworkParametersLIANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworks)
		ANNtf2_algorithmLIANN.defineNeuralNetworkParametersLIANN()
	elif(algorithm == "AEANN"):
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = ANNtf2_algorithmAEANN.defineTrainingParametersAEANN(dataset)
		numberOfLayers = ANNtf2_algorithmAEANN.defineNetworkParametersAEANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworks)
		ANNtf2_algorithmAEANN.defineNeuralNetworkParametersAEANN()
									
	#define epochs:
	
	if(trainMultipleFiles):
		fileIndexFirst = 0
		if(useSmallSentenceLengths):
			fileIndexLast = 11
		else:
			fileIndexLast = 1202

	noisySampleGeneration = False
	if(algorithm == "LREANN"):
		noisySampleGeneration, noisySampleGenerationNumSamples, noiseStandardDeviation = ANNtf2_algorithmLREANN.getNoisySampleGenerationNumSamples()
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
					trainDataList = ANNtf2_algorithmLREANN.generateTFtrainDataFromNParraysLREANN_expAUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses)
					exemplarDataList = ANNtf2_algorithmLREANN.generateTFexemplarDataFromNParraysLREANN_expAUANN(train_x, train_y, networkIndex, shuffleSize, batchSize, datasetNumClasses, generateClassTargetExemplars)
					test_y = ANNtf2_algorithmLREANN.generateYActualfromYLREANN_expAUANN(test_y, num_output_neurons)
					datasetNumClassTargets = datasetNumClasses
					datasetNumClasses = ANNtf2_algorithmLREANN.generateNumClassesActualLREANN_expAUANN(datasetNumClasses, num_output_neurons)
					exemplarDataListIterators = []
					for exemplarData in exemplarDataList:
						exemplarDataListIterators.append(iter(exemplarData))
				elif(algorithmLREANN == "LREANN_expCUANN"):
					trainDataList = ANNtf2_algorithmLREANN.generateTFtrainDataFromNParraysLREANN_expCUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses)
				elif(algorithmLREANN == "LREANN_expXUANN"):
					currentClassTarget = 0
					generateClassTargetExemplars = False
					if(e == 0):
						generateClassTargetExemplars = True
					trainDataList = ANNtf2_algorithmLREANN.generateTFtrainDataFromNParraysLREANN_expXUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses, generatePositiveSamples=True)
					datasetNumClassTargets = datasetNumClasses
					samplePositiveDataList = ANNtf2_algorithmLREANN.generateTFtrainDataFromNParraysLREANN_expXUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses, generatePositiveSamples=True)
					if(XUANNnegativeSamplesComplement):
						sampleNegativeDataList = ANNtf2_algorithmLREANN.generateTFtrainDataFromNParraysLREANN_expXUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses, generatePositiveSamples=False)					
					elif(XUANNnegativeSamplesAll):
						#implementation limitation (sample negative contains a selection of experiences from all classes, not just negative classes) - this simplification deemed valid under assumptions: calculations will be averaged over large negative batch and numberClasses >> 2
						sampleNegativeData = generateTFtrainDataFromNParrays(train_x, train_y, shuffleSize, batchSize)	#CHECKTHIS
						sampleNegativeDataList = []
						sampleNegativeDataList.append(sampleNegativeData)
					elif(XUANNnegativeSamplesRandom):
						sampleNegativeDataList = ANNtf2_algorithmLREANN.generateTFtrainDataFromNParraysLREANN_expXUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses, generatePositiveSamples=True)					
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
						batchYactual = ANNtf2_algorithmLREANN.generateTFYActualfromYandExemplarYLREANN_expAUANN(batchY, exemplarsY)
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
				
				for networkIndex in range(1, numberOfNetworks+1):

					if(algorithm == "ANN"):
						executeOptimisation(batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex)
						if(batchIndex % displayStep == 0):
							pred = neuralNetworkPropagation(batchX, networkIndex)
							#print("pred.shape = ", pred.shape)
							loss = calculateLossCrossEntropy(pred, batchY, datasetNumClasses, costCrossEntropyWithLogits)
							acc = calculateAccuracy(pred, batchY)
							print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
							predNetworkAverage = predNetworkAverage + pred
					elif(algorithm == "SANI"):
						#learning algorithm not yet implemented:
						if(batchSize > 1):
							pred = neuralNetworkPropagation(batchX)
						if(batchIndex % displayStep == 0):
							pred = neuralNetworkPropagation(batchX)
							#acc = tf.reduce_mean(tf.dtypes.cast(pred, tf.float32))
							#print("batchIndex: %i, accuracy: %f" % (batchIndex, acc))
					elif(algorithm == "LREANN"):
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
					elif(algorithm == "FBANN"):
						executeOptimisation(batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex)
						if(batchIndex % displayStep == 0):
							pred = neuralNetworkPropagation(batchX, networkIndex)
							#print("pred = ", pred)
							loss = calculateLossCrossEntropy(pred, batchY, datasetNumClasses, costCrossEntropyWithLogits)
							acc = calculateAccuracy(pred, batchY)
							print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
							predNetworkAverage = predNetworkAverage + pred
					elif(algorithm == "EIANN"):
						if(ANNtf2_algorithmEIANN.learningAlgorithmFinalLayerBackpropHebbian):
							#first learning algorithm: perform neuron independence training
							batchYoneHot = tf.one_hot(batchY, depth=datasetNumClasses)
							executeLearningEIANN(batchX, batchYoneHot, networkIndex)
							#second learning algorithm (final layer hebbian connections to output class targets):
						executeOptimisation(batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex)
						if(batchIndex % displayStep == 0):
							pred = neuralNetworkPropagation(batchX, networkIndex)
							#print("pred.shape = ", pred.shape)
							loss = calculateLossCrossEntropy(pred, batchY, datasetNumClasses, costCrossEntropyWithLogits)
							acc = calculateAccuracy(pred, batchY)
							print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
							predNetworkAverage = predNetworkAverage + pred
					elif(algorithm == "LIANN"):
						#first learning algorithm: perform neuron independence training
						batchYoneHot = tf.one_hot(batchY, depth=datasetNumClasses)
						executeLearningLIANN(batchX, batchYoneHot, networkIndex)
						if(ANNtf2_algorithmLIANN.learningAlgorithmFinalLayerBackpropHebbian):
							#second learning algorithm (final layer hebbian connections to output class targets):
							executeOptimisation(batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex)
							if(batchIndex % displayStep == 0):
								pred = neuralNetworkPropagation(batchX, networkIndex)
								#print("pred.shape = ", pred.shape)
								loss = calculateLossCrossEntropy(pred, batchY, datasetNumClasses, costCrossEntropyWithLogits)
								acc = calculateAccuracy(pred, batchY)
								print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
								predNetworkAverage = predNetworkAverage + pred
					elif(algorithm == "AEANN"):
						for l in range(1, numberOfLayers+1):
							executeOptimisation(batchX, batchY, datasetNumClasses, numberOfLayers, optimizer, networkIndex, l=l)
						if(batchIndex % displayStep == 0):
							pred = ANNtf2_algorithmAEANN.neuralNetworkPropagationAEANNtest(batchX, networkIndex)	#skip autoencoders
							#print("pred.shape = ", pred.shape)
							loss = calculateLossCrossEntropy(pred, batchY, datasetNumClasses, costCrossEntropyWithLogits)
							acc = calculateAccuracy(pred, batchY)
							print("networkIndex: %i, batchIndex: %i, loss: %f, accuracy: %f" % (networkIndex, batchIndex, loss, acc))
							predNetworkAverage = predNetworkAverage + pred
								
				if(algorithm == "LREANN"):
					if(algorithmLREANN == "LREANN_expAUANN"):
						#batchYactual = ANNtf2_algorithmLREANN.generateTFYActualfromYandExemplarYLREANN_expAUANN(batchY, exemplarsY)
						currentClassTarget = currentClassTarget+1
						if(currentClassTarget == datasetNumClassTargets):
							currentClassTarget = 0
						trainDataIndex = currentClassTarget
					elif(algorithmLREANN == "LREANN_expXUANN"):
						currentClassTarget = currentClassTarget+1
						if(currentClassTarget == datasetNumClassTargets):
							currentClassTarget = 0
						trainDataIndex = currentClassTarget
							
				if(batchIndex % displayStep == 0):
					if(trainMultipleNetworks):
						predNetworkAverage = predNetworkAverage / numberOfNetworks
						loss = calculateLossCrossEntropy(predNetworkAverage, batchYactual, datasetNumClasses, costCrossEntropyWithLogits)
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
				elif(algorithm == "LREANN"):
					pred = neuralNetworkPropagation(test_x, networkIndex)
					print("Test Accuracy: networkIndex: %i, %f" % (networkIndex, calculateAccuracy(pred, test_y)))
					predNetworkAverageAll = predNetworkAverageAll + pred
				elif(algorithm == "FBANN"):
					pred = neuralNetworkPropagation(test_x, networkIndex)
					print("Test Accuracy: networkIndex: %i, %f" % (networkIndex, calculateAccuracy(pred, test_y)))
					predNetworkAverageAll = predNetworkAverageAll + pred				
				elif(algorithm == "EIANN"):
					pred = neuralNetworkPropagation(test_x, networkIndex)	#test_x batch may be too large to propagate simultaneously and require subdivision
					print("Test Accuracy: networkIndex: %i, %f" % (networkIndex, calculateAccuracy(pred, test_y)))
					predNetworkAverageAll = predNetworkAverageAll + pred
				elif(algorithm == "LIANN"):
					pred = neuralNetworkPropagation(test_x, networkIndex)
					print("Test Accuracy: networkIndex: %i, %f" % (networkIndex, calculateAccuracy(pred, test_y)))
					predNetworkAverageAll = predNetworkAverageAll + pred
				elif(algorithm == "AEANN"):
					pred = ANNtf2_algorithmAEANN.neuralNetworkPropagationAEANNtest(test_x, networkIndex)	#skip autoencoders
					print("Test Accuracy: networkIndex: %i, %f" % (networkIndex, calculateAccuracy(pred, test_y)))
					predNetworkAverageAll = predNetworkAverageAll + pred
													
			if(trainMultipleNetworks):
					predNetworkAverageAll = predNetworkAverageAll / numberOfNetworks
					#print("predNetworkAverageAll", predNetworkAverageAll)
					acc = calculateAccuracy(predNetworkAverageAll, test_y)
					print("Test Accuracy: %f" % (acc))

if __name__ == "__main__":
	if(algorithm == "BAANN"):
		fileIndexTemp = 0
		learningRate, trainingSteps, batchSize, displayStep, numEpochs = ANNtf2_algorithmBAANN.defineTrainingParametersBAANN(dataset)
		numberOfFeaturesPerWord, paddingTagIndex, datasetNumFeatures, datasetNumClasses, datasetNumExamplesTemp, train_x, train_y, test_x, test_y = loadDataset(fileIndexTemp)
		ANNtf2_algorithmBAANN.BAANNmain(train_x, train_y, test_x, test_y, datasetNumFeatures, datasetNumClasses, batchSize, trainingSteps, numEpochs)
	else:
		main()
	
