# -*- coding: utf-8 -*-
"""ANNtf2_algorithmCANN_expBUANN.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Usage:
see ANNtf2.py

# Description

Define fully connected burst/spike update artificial neural network (CANN_expBUANN)

- Author: Richard Bruce Baxter - Copyright (c) 2020-2021 Baxter AI (baxterai.com)

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *
import ANNtf2_operations
import ANNtf2_globalDefs
import math
from numpy import random

#
# BUANN biological implementation requirements:
#
# backpropagation approximation notes:
# error_l = (W_l+1 * error_l+1) * A_l
# dC/dB = error_l
# dC/dW = A_l-1 * error_l
#
# backpropagation error is stored in temporary firing rate modification [increase/decrease] of neurons (ie Atrace->Aideal)
# Aerror_l update is applied based on signal pass through (W), higher level temporary firing rate adjustment, and current firing rate. error_l = (W_l+1 * error_l+1) * A_l
# W_l update is applied based on firing rate of lower layer and higher level temporary firing rate adjustment. dC/dW = A_l-1 * error_l
#
# BUANN approximates backpropagation for constrained/biological assumptions
# Error calculations are achieved by repropagating signal through neuron and measuring either a) temporary modulation in output (Aideal) relative to original (Atrace), or b) output of a specific error storage neurotransmitter receptor
#
# Outstanding Biological Requirement: Need to identify a method to pass (single layer) error signal back through neuron from tip of axon to base of dendrite (internal/external signal?)
#	the original BUANN (learningAlgorithm == "backpropApproximation3/backpropApproximation4") attempts to achieve this by sending a trial +/- signal from the lower layer l neuron k and slowly ramping it up/down (increasing/decreasing its effective error) until the above layer l+1 neurons reach their ideal values/errors  
#

debugOnlyTrainFinalLayer = False	#debug weight update method only (not Aideal calculation method)	#requires recalculateAtraceUnoptimisedBio==False
debugVerboseOutput = False
debugVerboseOutputTrain = False

averageAerrorAcrossBatch = True	#BUANN was originally implemented to calculate independent idealA for each batch index (rather than averaged across batch)

errorImplementationAlgorithm = "storeErrorAsModulationOfSignalPropagationNeurotransmitterReceptor"	#original	#a) modulates primary propagation neurotransmitter receptor (+/-) to store l error, and for the calculation of l-1 error
#errorImplementationAlgorithm = "storeErrorAsModulationOfUniqueNeurotransmitterReceptor"	#b) designates a specific neurotransmitter receptor to store l error, and for the calculation of l-1 error

#learning algorithm variants in order of emulation similarity to formal backpropagation:
learningAlgorithm = "backpropApproximation1"
#learningAlgorithm = "backpropApproximation2"
#learningAlgorithm = "backpropApproximation3"
#learningAlgorithm = "backpropApproximation4"	#original proposal	#emulates backpropagation using a variety of shortcuts (with optional thresholding), but does not emulate backPropagation completely - error_l (Aideal_l) calculations are missing *error_l+1 (multiply by the strength of the higher layer error)
#learningAlgorithm = "backpropApproximation5"	#simplifies BUANN algorithm to only consider +/- performance (not numerical/weighted performance)
	#probably only feasible with useBinaryWeights
	#note if useBinaryWeights then could more easily biologically predict the effect of adjusting Aideal of lower layer neuron k on performance of upper layer (perhaps without even trialling the adjustment)

#errorStorageAlgorithm = "useAerror"	#l+1 error is stored as a linear modulation of post synaptic receptor
#errorStorageAlgorithm = "useAideal" 	#original	#l+1 error is stored as a hypothetical difference between Atrace and Aideal [ratio]

if(learningAlgorithm == "backpropApproximation1"):
	#strict is used for testing only	#no known biological implementation
	#requires recalculateAtraceUnoptimisedBio==False
	errorStorageAlgorithm = "useAerror"
elif(learningAlgorithm == "backpropApproximation2"):
	#requires recalculateAtraceUnoptimisedBio==False
	errorStorageAlgorithm = "useAerror"
elif(learningAlgorithm == "backpropApproximation3"):
	errorStorageAlgorithm = "useAerror"
elif(learningAlgorithm == "backpropApproximation4"):
	errorStorageAlgorithm = "useAideal"
	useWeightUpdateDirectionHeuristicBasedOnExcitatoryInhibitorySynapseType = True	#enables rapid weight updates, else use stocastic (test both +/-) weight upates
	useMultiplicationRatherThanAdditionOfDeltaValues = True	#CHECKTHIS #this ensures that Aideal/weight updates are normalised across their local layer (to minimise the probability an alternate class data propagation will be interferred with by the update)
elif(learningAlgorithm == "backpropApproximation5"):
	errorStorageAlgorithm = "useAideal"
	useWeightUpdateDirectionHeuristicBasedOnExcitatoryInhibitorySynapseType = True
	useMultiplicationRatherThanAdditionOfDeltaValues = False
	

activationFunctionType = "sigmoid"	#default
#activationFunctionType = "softmax"	#trial only
#activationFunctionType = "relu"	#not currently supported; a) cannot converge with relu function at final layer, b) requires loss function

applyFinalLayerLossFunction = False		#if False: normalise the error calculation across all layers, taking y_target as Aideal of top layer
if(learningAlgorithm == "backpropApproximation1"):
	activationFunctionTypeFinalLayer = "softmax"
	applyFinalLayerLossFunction = True
elif(learningAlgorithm == "backpropApproximation2"):
	activationFunctionTypeFinalLayer = "softmax"
	applyFinalLayerLossFunction = True
else:
	activationFunctionTypeFinalLayer = "sigmoid"	#default	#doesn't currently converge with final layer loss function calculated based on sigmoid
	applyFinalLayerLossFunction = True	
	
errorFunctionTypeDelta = True
errorFunctionTypeDeltaFinalLayer = True		#sigmoid/softmax has already been calculated [Aideal has been created for final layer] so can simply extrace delta error here 	#OLD: use sigmoid/softmax loss for final layer - consider using more simply delta loss here

updateOrder = "updateWeightsAfterAidealCalculations"	#method 1
#updateOrder = "updateWeightsDuringAidealCalculations"	#method 2
#updateOrder = "updateWeightsBeforeAidealCalculations"	#method 3

takeAprevLayerFromTraceDuringWeightUpdates = True	
	#this parameter value should not be critical to BUANN algorithm (it is currently set based on availability of Aideal of lower layer - ie if it has been precalculated)
	#difference between Aideal and Atrace of lower layer should be so small takeAprevLayerFromTraceDuringWeightUpdates shouldn't matter

recalculateAtraceUnoptimisedBio = False


if(not applyFinalLayerLossFunction):
	topLayerIdealAstrict = True #top level learning target (idealA) == y, else learning target (idealA) == A + deltaA
	topLayerIdealAproximity = 0.01	#maximum learning rate (effective learning rate will be less than this)

	
if(learningAlgorithm == "backpropApproximation4"):

	useMultiplicationRatherThanAdditionOfDeltaValuesAideal = False
	useMultiplicationRatherThanAdditionOfDeltaValuesW = False
	
	if(useMultiplicationRatherThanAdditionOfDeltaValues):
		useMultiplicationRatherThanAdditionOfDeltaValuesAideal = True
		useMultiplicationRatherThanAdditionOfDeltaValuesW = True
	else:
		useMultiplicationRatherThanAdditionOfDeltaValuesAideal = False
		useMultiplicationRatherThanAdditionOfDeltaValuesW = False
	learningRateMinFraction = 0.1	#minimum learning rate can be set to always be above 0 (learningRateMinFraction = fraction of learning rate)
	
	applyMinimiumAdeltaContributionThreshold = False 	#only adjust Aideal_k of l based on Aideal of l+1 if it significantly improves Aideal of l+1, where k is neuron index of l
	if(applyMinimiumAdeltaContributionThreshold):
		minimiumAdeltaContributionThreshold = 0.1	#fraction relative to original performance difference
		#minimiumAdeltaContributionThreshold = 1.0	#this contribution threshold is normalised wrt number of neurons (k) on l+1. default=1.0: if a Aideal_k adjustment on l contributes less than what on average an Aideal_k adjustment must necessarily contribute to achieve Aideal on l+1, then do not adjust Aideal_k (leave same as A_k)

	applySubLayerIdealAmultiplierRequirement = True
	if(applySubLayerIdealAmultiplierRequirement):
		subLayerIdealAmultiplierRequirement = 1.5 #idealA of each neuron k on l will only be adjusted if its modification achieves at least xM performance increase for Aideal on l+1
		applySubLayerIdealAmultiplierCorrection = True	#optional: adjust learning neuron learning based on performance multiplier
	else:
		applySubLayerIdealAmultiplierCorrection = False

if(learningAlgorithm == "backpropApproximation5"):
	subLayerIdealAlearningRateBase = 0.001	#small number used to ensure (reduce probablity) that update does not affect nonlinearity of signal upwards
else:
	subLayerIdealAlearningRateBase = 0.01	#each neuron k on l will be adjusted only by this amount (modified by its multiplication effect on Aideal of l+1)

debugWexplosion = False
debugFastTrain = False
if(debugFastTrain):
	learningRate = 0.01
else:
	learningRate = 0.001


useBatch = True
if(useBatch):
	if(debugFastTrain):
		batchSize = 1000
	else:
		batchSize = 100	#100
else:
	batchSize = 1	

biologicalConstraints = False	#batchSize=1, _?

sparsityLevel = 1.0	#probability of initial strong neural connection per neuron in layer

noisySampleGeneration = False
noisySampleGenerationNumSamples = 0
noiseStandardDeviation = 0

if(biologicalConstraints):
	useBinaryWeights = True	#increases stochastically updated training speed, but reduces final accuracy
	if(useBinaryWeights):	
		averageTotalInput = -1
		useBinaryWeightsReduceMemoryWithBool = False	#can use bool instead of float32 to limit memory required, but requires casting to float32 for matrix multiplications
	if(not useBatch):
		noisySampleGeneration = False	#possible biological replacement for input data batchSize > 1 (provides better performance than standard input data batchSize == 1, but less performance than input data batchSize > 10+)
		if(noisySampleGeneration):
			noisySampleGenerationNumSamples = 10
			noiseStandardDeviation = 0.03
else:
	useBinaryWeights = False

	
	

W = {}
B = {}

Wbackup = {}
Bbackup = {}

NETWORK_PARAM_INDEX_TYPE = 0
NETWORK_PARAM_INDEX_LAYER = 1
NETWORK_PARAM_INDEX_H_CURRENT_LAYER = 2
NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER = 3
NETWORK_PARAM_INDEX_VARIATION_DIRECTION = 4

if(not recalculateAtraceUnoptimisedBio):
	Atrace = {}
	
if(errorStorageAlgorithm == "useAideal"):
	Aideal = {}
elif(errorStorageAlgorithm == "useAerror"):
	Aerror = {}			

#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0
datasetNumClasses = 0

#randomNormal = tf.initializers.RandomNormal()

# Stochastic gradient descent optimizer.
optimizer = tf.optimizers.SGD(learningRate)
	
def getNoisySampleGenerationNumSamples():
	return noisySampleGeneration, noisySampleGenerationNumSamples, noiseStandardDeviation
	
def defineTrainingParametersCANN(dataset, trainMultipleFiles):
	
	if(trainMultipleFiles):
		if(dataset == "POStagSequence"):
			trainingSteps = 10000
		elif(dataset == "SmallDataset"):
			if(debugFastTrain):
				trainingSteps = 1000
			else:
				trainingSteps = 10000
		numEpochs = 10
	else:
		if(dataset == "POStagSequence"):
			trainingSteps = 10000
		elif(dataset == "SmallDataset"):
			if(debugFastTrain):
				trainingSteps = 1000
			else:
				trainingSteps = 10000
		if(useBatch):
			numEpochs = 100	#10
		else:
			numEpochs = 100
	
	displayStep = 100

	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	

def defineNetworkParametersCANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	global datasetNumClasses
	
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet)

	return numberOfLayers

def defineNeuralNetworkParametersCANN():
	
	tf.random.set_seed(5);
	if(useBinaryWeights):
		if(useBinaryWeightsReduceMemoryWithBool):
			dtype=tf.dtypes.bool
		else:
			dtype=tf.dtypes.float32
	else:
		#randomNormal = tf.initializers.RandomNormal()
		dtype=tf.dtypes.float32
	
	for networkIndex in range(1, numberOfNetworks+1):
	
		for l in range(1, numberOfLayers+1):

			if(useBinaryWeights):
				Wint = tf.random.uniform([n_h[l-1], n_h[l]], minval=0, maxval=2, dtype=tf.dtypes.int32)		#The lower bound minval is included in the range, while the upper bound maxval is excluded.
				W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(tf.dtypes.cast(Wint, dtype=dtype))
				#print("W[generateParameterNameNetwork(networkIndex, l, W)] = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
			else:
				W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(tf.random.normal([n_h[l-1], n_h[l]], stddev=sparsityLevel, dtype=dtype))		#tf.Variable(randomNormal([n_h[l-1], n_h[l]]))	
				#note stddev=sparsityLevel: a weakly tailed distribution for sparse activated network (such that the majority of weights are close to zero)
			B[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(tf.zeros(n_h[l], dtype=dtype))
	
			Wbackup[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(W[generateParameterNameNetwork(networkIndex, l, "W")])
			Bbackup[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(B[generateParameterNameNetwork(networkIndex, l, "B")])
		
			#Aerror = Aideal - Atrace
			if(errorStorageAlgorithm == "useAideal"):
				if(averageAerrorAcrossBatch):
					Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")] = tf.Variable(tf.zeros(n_h[l], dtype=tf.dtypes.float32))
				else:
					Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.dtypes.float32))			
			elif(errorStorageAlgorithm == "useAerror"):
				if(averageAerrorAcrossBatch):
					Aerror[generateParameterNameNetwork(networkIndex, l, "Aerror")] = tf.Variable(tf.zeros(n_h[l], dtype=tf.dtypes.float32))			
				else:
					Aerror[generateParameterNameNetwork(networkIndex, l, "Aerror")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.dtypes.float32))
					
			if(not recalculateAtraceUnoptimisedBio):
				if(averageAerrorAcrossBatch):
					Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")] = tf.Variable(tf.zeros(n_h[l], dtype=tf.dtypes.float32))
				else:
					Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.dtypes.float32))

	

def neuralNetworkPropagationCANN(x, networkIndex=1, recordAtrace=False):
	pred, A, Z = neuralNetworkPropagationCANNlayer(x, lTrain=numberOfLayers, networkIndex=networkIndex)
	return pred
	
def neuralNetworkPropagationCANNlayer(x, lTrain, networkIndex=1, recordAtrace=False):
			
	AprevLayer = x
	
	if(recordAtrace):
		Aaveraged = AprevLayer
		if(averageAerrorAcrossBatch):
			Aaveraged = tf.reduce_mean(Aaveraged, axis=0)      #average across all batches
		Atrace[generateParameterNameNetwork(networkIndex, 0, "Atrace")] = Aaveraged	#CHECKTHIS
	
	for l in range(1, lTrain+1):
	
		if(debugVerboseOutput):
			print("l = " + str(l))
			print("W = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
		
		A, Z = neuralNetworkPropagationCANNlayerL(AprevLayer, l, networkIndex)

		if(recordAtrace):
			Aaveraged = A
			if(averageAerrorAcrossBatch):
				Aaveraged = tf.reduce_mean(Aaveraged, axis=0)      #average across all batches
			Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")] = Aaveraged
		
		AprevLayer = A
		
	pred = tf.nn.softmax(Z)
		
	return pred, A, Z

def neuralNetworkPropagationCANNlayerL(AprevLayer, l, networkIndex=1):

	if(useBinaryWeights):
		if(useBinaryWeightsReduceMemoryWithBool):
			Wfloat = tf.dtypes.cast(W[generateParameterNameNetwork(networkIndex, l, "W")], dtype=tf.float32)
			Bfloat = tf.dtypes.cast(B[generateParameterNameNetwork(networkIndex, l, "B")], dtype=tf.float32)
			Z = tf.add(tf.matmul(AprevLayer, Wfloat), Bfloat)
		else:
			Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
		A = activationFunction(Z, n_h[l-1])
	else:
		Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
		A = activationFunction(Z)
	
	return A, Z
			
			
def neuralNetworkPropagationCANN_test(x, y, networkIndex=1):

	pred = neuralNetworkPropagationCANN(x, networkIndex)
	loss = ANNtf2_operations.crossEntropy(pred, y, datasetNumClasses, costCrossEntropyWithLogits=False)
	acc = ANNtf2_operations.calculateAccuracy(pred, y)
	
	return loss, acc
	

def neuralNetworkPropagationCANN_expBUANNtrain(x, y, networkIndex=1):
	
	#print("numberOfLayers = ", numberOfLayers)
	
	if(debugOnlyTrainFinalLayer):
		minLayerToTrain = numberOfLayers
	else:
		minLayerToTrain = 1	#do not calculate Aideal for input layer as this is always set to x
	
	#1. initial propagation;
	y_true = tf.one_hot(y, depth=datasetNumClasses)
	pred, A, Z = neuralNetworkPropagationCANNlayer(x, numberOfLayers, networkIndex, recordAtrace=(not recalculateAtraceUnoptimisedBio))
	
	if(activationFunctionTypeFinalLayer == "sigmoid"):
		y_pred = A	#A is after sigmoid
	elif(activationFunctionTypeFinalLayer == "softmax"):	
		y_pred = pred	#pred is after softmax

	#2./3. calculate Aideal / W updates;
	
	calculateAerrorTopLayer(y_pred, y_true, networkIndex)			
	calculateAerrorBottomLayer(x, minLayerToTrain, networkIndex)		
	
	if(updateOrder == "updateWeightsAfterAidealCalculations"):
		for l in reversed(range(minLayerToTrain, numberOfLayers)):
			if(debugVerboseOutputTrain):
				print("calculateAerror: l = ", l)
			calculateAerror(l, networkIndex)	
		for l in range(minLayerToTrain, numberOfLayers+1):	#optimisation note: this can be done in parallel (weights can be updated for each layer simultaneously)
			#print("updateWeightsBasedOnAerror: l = ", l)
			updateWeightsBasedOnAerror(l, x, y, networkIndex)
	elif(updateOrder == "updateWeightsDuringAidealCalculations"):
		for l in reversed(range(minLayerToTrain, numberOfLayers+1)):	#2: do not calculate Aideal for input layer as this is always set to x
			if(debugVerboseOutputTrain):
				print("calculateAerror: l = ", l)
			if(l != 1):
				calculateAerror(l-1, networkIndex)	
			updateWeightsBasedOnAerror(l, x, y, networkIndex)
	elif(updateOrder == "updateWeightsBeforeAidealCalculations"):
		for l in reversed(range(minLayerToTrain, numberOfLayers+1)):
			if(debugVerboseOutputTrain):
				print("calculateAerror: l = ", l)
			updateWeightsBasedOnAerror(l, x, y, networkIndex)
			if(l != 1):
				calculateAerror(l-1, networkIndex)


def calculateAerrorTopLayer(y_pred, y_true, networkIndex=1):
	
	if(applyFinalLayerLossFunction):
		if(activationFunctionTypeFinalLayer == "softmax"):
			loss = ANNtf2_operations.crossEntropy(y_pred, y_true, datasetNumClasses, costCrossEntropyWithLogits=False, oneHotEncoded=True, reduceMean=False)
		elif(activationFunctionTypeFinalLayer == "sigmoid"):		
			loss = ANNtf2_operations.crossEntropy(y_pred, y_true, datasetNumClasses=None, costCrossEntropyWithLogits=True, reduceMean=False)	#loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
		else:
			print("activationFunctionTypeFinalLayer not currently supported by BUANN = ", activationFunctionTypeFinalLayer)
			exit()
		AerrorAbs = loss
	
		#calculate signed error:
		AidealDelta = calculateADelta(y_true, y_pred)
		AidealDeltaSign = tf.sign(AidealDelta)
		AerrorVec = tf.multiply(AerrorAbs, AidealDeltaSign)	#updateWeightsBasedOnAidealHeuristic requires directional error
		
		#print("AerrorVec = ", AerrorVec)
		#print("AerrorAbs = ", AerrorAbs)
	else:
		if(topLayerIdealAstrict):
			AerrorVec = tf.subtract(y_true, y_pred)	
		else:
			#calculate Aideal of final layer based on y	
			AdeltaMax = tf.subtract(y_true, y_pred)	
			AerrorVec = calculateDeltaTF(AdeltaMax, topLayerIdealAproximity, True, applyMinimia=False)

	if(averageAerrorAcrossBatch):
		AerrorVec = tf.reduce_mean(AerrorVec, axis=0)      #average across all batches 
		y_pred = tf.reduce_mean(y_pred, axis=0)      #average across all batches 
		
	setAerror(AerrorVec, y_pred, numberOfLayers, networkIndex)
	
	#print("AerrorVec = ", AerrorVec)
	#print("y_pred = ", y_pred)
			
				
def calculateAerrorBottomLayer(x, minLayerToTrain, networkIndex=1):
	
	if(averageAerrorAcrossBatch):
		xAveraged = tf.reduce_mean(x, axis=0)      #average across all batches
	setAerrorGivenAideal(xAveraged, xAveraged, 0, networkIndex)	#set Aideal of input layer to x
	
	if(debugOnlyTrainFinalLayer):
		for l in range(1, minLayerToTrain):
			setAerrorGivenAideal(Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")], Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")], l, networkIndex)	#set Aideal of input layer to Atrace


def calculateAerror(l, networkIndex=1):

	#stocastically identify Aideal of l (lower) based on Aideal of l+1
		#this is biologically achieved by temporarily/independently adjusting the firing rate (~bias) of each neuron (index k) on l, and seeing if this better achieves Aideal of l+1
		#feedback (positive/negtive trial) is given from higher level l+1 to l_k in the form of "simple" [boolean] ~local chemical signal

	A = Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")]	#get original A value of lower layer
	if(learningAlgorithm == "backpropApproximation1"):
		setAerrorBackpropStrict(A, l, networkIndex)
	else:
		for k in range(n_h[l]):
			if(debugVerboseOutputTrain):
				print("\tcalculateAideal: k = ", k)
			#try both positive and negative adjustments of A_l_k;
			if(learningAlgorithm == "backpropApproximation2"):
				setAerrorBackpropSemi(A, k, l, networkIndex)
			else:
				trialAidealMod(True, A, k, l, networkIndex)
				trialAidealMod(False, A, k, l, networkIndex)

def setAerrorBackpropSemi(A, k, l, networkIndex):

	#error_l = (W_l+1 * error_l+1) * A_l = (A_l*W_l+1) * Aideal_l+1 - (A_l*W_l+1) * Atrace_l+1 

	AlayerAboveWithError = trialAerrorMod(True, A, k, l, networkIndex)
	AlayerAboveWithoutError = trialAerrorMod(False, A, k, l, networkIndex)
	AerrorLayer = calculateErrorAtrial(AlayerAboveWithoutError, AlayerAboveWithError, networkIndex, averageType="none")
	if(averageAerrorAcrossBatch):
		AerrorLayer = tf.reduce_mean(AidealDelta, axis=0)   #average across all k neurons on l+1
	else:
		AerrorLayer = tf.reduce_mean(AidealDelta, axis=1)   #average across all k neurons on l+1
		
	setAerror(AerrorLayer, Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")], l, networkIndex)
	
	#FUTURE: only set Aerror of neuron k (especially required if error signal is transferred externally rather than internally)

def trialAerrorMod(applyAboveLayerError, A, k, l, networkIndex):

	AtrialAbove, ZtrialAbove = neuralNetworkPropagationCANNlayerL(A, l+1, networkIndex)
	
	if(applyAboveLayerError):
		AerrorAbove = Aerror[generateParameterNameNetwork(networkIndex, l+1, "Aerror")]
		AerrorLayer = tf.multiply(ZtrialAbove, AerrorAbove)	
	else:
		AerrorLayer = ZtrialAbove
		
	return AerrorLayer

def setAerrorBackpropStrict(A, l, networkIndex):
	AerrorVec = calculateAerrorBackpropStrict(A, l, networkIndex)
	setAerror(AerrorVec, Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")], l, networkIndex)
	
def calculateAerrorBackpropStrict(A, l, networkIndex):
	AerrorAbove = Aerror[generateParameterNameNetwork(networkIndex, l+1, "Aerror")]
	WAbove = W[generateParameterNameNetwork(networkIndex, l+1, "W")]
	
	if(averageAerrorAcrossBatch):
		AerrorAbove = tf.expand_dims(AerrorAbove, axis=0)

	AerrorVec = tf.matmul(AerrorAbove, tf.transpose(WAbove))	#(W_l+1 * error_l+1)	#multiply by the strength of the signal weight passthrough	#multiply by the strength of the higher layer error	
	
	if(averageAerrorAcrossBatch):
		AerrorVec = tf.squeeze(AerrorVec)
		
	AerrorVec = tf.multiply(AerrorVec, A) #* A_l	#multiply by the strength of the current layer signal
	return AerrorVec

def trialAerrorMod(direction, A, k, l, networkIndex):

	if(direction):
		trialAmodValue = subLayerIdealAlearningRateBase
	else:
		trialAmodValue = -subLayerIdealAlearningRateBase
	
	columnsIdx = tf.constant([k])
	AK = tf.gather(A, columnsIdx, axis=1)	#Atrial[:,k]	
	AtrialK = AK
	if(learningAlgorithm == "backpropApproximation5"):
		AtrialKdelta = trialAmodValue
	elif(learningAlgorithm == "backpropApproximation4"):
		AtrialKdelta = calculateDeltaTF(AtrialK, trialAmodValue, useMultiplicationRatherThanAdditionOfDeltaValuesAideal)	#this integrates the fact in backpropagation Aerror should be linearly dependent on A  #* A_l	#multiply by the strength of the current layer signal
	elif(learningAlgorithm == "backpropApproximation3"):
		AtrialKdelta = trialAmodValue
	AtrialK = tf.add(AtrialK, AtrialKdelta)
	
	Atrial = A
	Atrial = modifyTensorRowColumn(Atrial, False, k, AtrialK, isVector=True)	#Atrial[:,k] = (trialAmodValue)

	AtrialAbove, ZtrialAbove = neuralNetworkPropagationCANNlayerL(Atrial, l+1, networkIndex)
	successfulTrial, performanceMultiplier = testAtrialPerformance(AtrialKdelta, AtrialAbove, l, networkIndex)
	successfulTrialFloat = tf.dtypes.cast(successfulTrial, dtype=tf.float32)
	
	if(learningAlgorithm == "backpropApproximation4"):
		if(applySubLayerIdealAmultiplierCorrection):
			AtrialKdelta = tf.multiply(AtrialKdelta, performanceMultiplier)	#W_l+1	#multiply by the strength of the signal weight passthrough
	elif(learningAlgorithm == "backpropApproximation3"):
		# error_l = (W_l+1 * error_l+1) * A_l
		AtrialKdelta = tf.multiply(AtrialKdelta, performanceMultiplier)	#W_l+1	#multiply by the strength of the signal weight passthrough
		AtrialKdelta = tf.multiply(AtrialKdelta, AtrialK) #* A_l	#multiply by the strength of the current layer signal

	AtrialKdeltaSuccessful = tf.multiply(AtrialKdelta, successfulTrialFloat)
	AtrialKSuccessful = tf.add(AtrialK, AtrialKdeltaSuccessful)
	
	if(debugVerboseOutputTrain):
		print("AtrialKSuccessful", AtrialKSuccessful)

	Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")] = modifyTensorRowColumn(Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")], False, k, AtrialKSuccessful, isVector=True)

	return AlayerAbove
	




def trialAidealMod(direction, A, k, l, networkIndex):

	if(direction):
		trialAmodValue = subLayerIdealAlearningRateBase
	else:
		trialAmodValue = -subLayerIdealAlearningRateBase
	
	columnsIdx = tf.constant([k])
	AK = tf.gather(A, columnsIdx, axis=1)	#Atrial[:,k]	
	AtrialK = AK
	if(learningAlgorithm == "backpropApproximation5"):
		AtrialKdelta = trialAmodValue
	elif(learningAlgorithm == "backpropApproximation4"):
		AtrialKdelta = calculateDeltaTF(AtrialK, trialAmodValue, useMultiplicationRatherThanAdditionOfDeltaValuesAideal)	#this integrates the fact in backpropagation Aerror should be linearly dependent on A  #* A_l	#multiply by the strength of the current layer signal
	elif(learningAlgorithm == "backpropApproximation3"):
		AtrialKdelta = trialAmodValue
	AtrialK = tf.add(AtrialK, AtrialKdelta)
	
	Atrial = A
	Atrial = modifyTensorRowColumn(Atrial, False, k, AtrialK, isVector=True)	#Atrial[:,k] = (trialAmodValue)

	AtrialAbove, ZtrialAbove = neuralNetworkPropagationCANNlayerL(Atrial, l+1, networkIndex)
	successfulTrial, performanceMultiplier = testAtrialPerformance(AtrialKdelta, AtrialAbove, l, networkIndex)
	successfulTrialFloat = tf.dtypes.cast(successfulTrial, dtype=tf.float32)
	
	if(learningAlgorithm == "backpropApproximation4"):
		if(applySubLayerIdealAmultiplierCorrection):
			AtrialKdelta = tf.multiply(AtrialKdelta, performanceMultiplier)	#W_l+1	#multiply by the strength of the signal weight passthrough
	elif(learningAlgorithm == "backpropApproximation3"):
		# error_l = (W_l+1 * error_l+1) * A_l
		AtrialKdelta = tf.multiply(AtrialKdelta, performanceMultiplier)	#W_l+1	#multiply by the strength of the signal weight passthrough
		AtrialKdelta = tf.multiply(AtrialKdelta, AtrialK) #* A_l	#multiply by the strength of the current layer signal

	AtrialKdeltaSuccessful = tf.multiply(AtrialKdelta, successfulTrialFloat)
	AtrialKSuccessful = tf.add(AtrialK, AtrialKdeltaSuccessful)
	
	if(debugVerboseOutputTrain):
		print("AtrialKSuccessful", AtrialKSuccessful)
	
	Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")] = modifyTensorRowColumn(Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")], False, k, AtrialKSuccessful, isVector=True)


def testAtrialPerformance(AtrialKdelta, AtrialAbove, l, networkIndex):
		
	performanceMultiplier = None	#performanceMultiplier: this calculates the ratio of the performance gain relative to the lower layer adjustment
	
	successfulTrial, trialPerformanceGain = testAtrialPerformanceAbove(AtrialAbove, l+1, networkIndex)
	successfulTrial = tf.expand_dims(successfulTrial, axis=1)
	
	if(learningAlgorithm == "backpropApproximation4"):
		trialPerformanceGain = tf.expand_dims(trialPerformanceGain, axis=1)
		performanceMultiplier = tf.divide(trialPerformanceGain, tf.abs(AtrialKdelta))	#added tf.abs to ensure sign of performanceMultiplier is maintained	#OLD: fix this; sometimes divides by zero	
			
		if(applySubLayerIdealAmultiplierRequirement):
			performanceMultiplierSuccessful = tf.greater(performanceMultiplier, subLayerIdealAmultiplierRequirement)
			successfulTrial = tf.logical_and(successfulTrial, performanceMultiplierSuccessful)
	elif(learningAlgorithm == "backpropApproximation3"):
		trialPerformanceGain = tf.expand_dims(trialPerformanceGain, axis=1)
		performanceMultiplier = tf.divide(trialPerformanceGain, tf.abs(AtrialKdelta))
		
	return successfulTrial, performanceMultiplier
	

def testAtrialPerformanceAbove(AtrialAbove, l, networkIndex):
	
	#print("l = ", l )
	#print("1 Aideal.shape = ", Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")].shape)
	
	AidealDeltaOrig = calculateAidealDelta(Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")], l, networkIndex)
	AidealDeltaTrial = calculateAidealDelta(AtrialAbove, l, networkIndex)
	
	AidealDeltaOrigAvg = tf.reduce_mean(AidealDeltaOrig, axis=1)   #average across all k neurons on l
	AidealDeltaTrialAvg = tf.reduce_mean(AidealDeltaTrial, axis=1) #average across all k neurons on l
	AidealDeltaOrigAvgAbs = tf.abs(AidealDeltaOrigAvg)
	AidealDeltaTrialAvgAbs = tf.abs(AidealDeltaTrialAvg)
	successfulTrial = tf.less(AidealDeltaTrialAvgAbs, AidealDeltaOrigAvgAbs)
	
	if(learningAlgorithm == "backpropApproximation5"):
		trialPerformanceGain = None
	elif(learningAlgorithm == "backpropApproximation4"):
		#apply thresholding
		if(applyMinimiumAdeltaContributionThreshold):
			AidealDeltaOrigAvgThreshold = tf.multiply(tf.sign(AidealDeltaOrigAvg), tf.subtract(tf.abs(AidealDeltaOrigAvg), minimiumAdeltaContributionThreshold))
				#OLD: AidealDeltaOrigAvgThreshold = tf.multiply(tf.sign(AidealDeltaOrigAvg), tf.subtract(tf.abs(AidealDeltaOrigAvg), minimiumAdeltaContributionThreshold)) #tf.maximum(tf.subtract(tf.abs(AidealDeltaOrigAvg), minimiumAdeltaContributionThreshold), 0.0)
			successfulTrial = tf.math.logical_not(tf.math.logical_xor(tf.less(AidealDeltaTrialAvg, AidealDeltaOrigAvgThreshold), tf.equal(tf.sign(AidealDeltaTrialAvg), 1)))    #tf.multiply(tf.less(AidealDeltaTrialAvg, AidealDeltaOrigAvg), tf.sign(AidealDeltaTrialAvg))	
		else:
			AidealDeltaOrigAvgAbs = tf.abs(AidealDeltaOrigAvg)
			AidealDeltaTrialAvgAbs = tf.abs(AidealDeltaTrialAvg)
			successfulTrial = tf.less(AidealDeltaTrialAvgAbs, AidealDeltaOrigAvgAbs)			
		trialPerformanceGain = tf.multiply(tf.subtract(AidealDeltaOrigAvg, AidealDeltaTrialAvg), tf.sign(AidealDeltaTrialAvg))	#orig trialPerformanceGain calculation method 	#W_l+1	#multiply by the strength of the signal weight passthrough
		#Algorithm limitation - Missing:  * error_l+1	#multiply by the strength of the higher layer error
	elif(learningAlgorithm == "backpropApproximation3"):
		trialPerformanceGain = tf.multiply(tf.subtract(AidealDeltaOrig, AidealDeltaTrial), tf.sign(AidealDeltaTrial))	#W_l+1	#multiply by the strength of the signal weight passthrough
		trialPerformanceGain = tf.multiply(trialPerformanceGain, AidealDeltaOrig)	# * error_l+1	#multiply by the strength of the higher layer error
		trialPerformanceGain = tf.reduce_mean(trialPerformanceGain, axis=1) #average across all k neurons on l
			
	return successfulTrial, trialPerformanceGain

		
		
def updateWeightsBasedOnAerror(l, x, y, networkIndex):
	
	
	if(learningAlgorithm == "backpropApproximation1"):
		Wlayer = W[generateParameterNameNetwork(networkIndex, l, "W")]
		Blayer = B[generateParameterNameNetwork(networkIndex, l, "B")]
		
		AtraceBelow = Atrace[generateParameterNameNetwork(networkIndex, l-1, "Atrace")]
		AerrorLayer = Aerror[generateParameterNameNetwork(networkIndex, l, "Aerror")]	
		
		#print("AerrorLayer.shape orig = ", AerrorLayer.shape)
		
		if(not averageAerrorAcrossBatch):
			AtraceBelow = tf.reduce_mean(AtraceBelow, axis=0)      #average across all batches 
			AerrorLayer = tf.reduce_mean(AerrorLayer, axis=0)      #average across all batches 
			
		#print("AtraceBelow.shape = ", AtraceBelow.shape)
		#print("AerrorLayer.shape = ", AerrorLayer.shape)
		
		AtraceBelow = tf.expand_dims(AtraceBelow, axis=1)	#required for matmul preparation
		AerrorLayer = tf.expand_dims(AerrorLayer, axis=0)	#required for matmul preparation
			
		Wdelta = tf.matmul(AtraceBelow, AerrorLayer)	# dC/dW = A_l-1 * error_l
		Bdelta = AerrorLayer	# dC/dB = error_l
		
		Wlayer = tf.add(Wlayer, tf.multiply(Wdelta, learningRate))
		Blayer = tf.add(Blayer, tf.multiply(Bdelta, learningRate))
		
		W[generateParameterNameNetwork(networkIndex, l, "W")] = Wlayer
		B[generateParameterNameNetwork(networkIndex, l, "B")] = Blayer
	elif(learningAlgorithm == "backpropApproximation2"):
		print("updateWeightsBasedOnAerror warning: learningAlgorithm == backpropApproximation2 has not been coded")
	elif(learningAlgorithm == "backpropApproximation3"):
		print("updateWeightsBasedOnAerror warning: learningAlgorithm == backpropApproximation3 has not been coded")
	else:
		if(takeAprevLayerFromTraceDuringWeightUpdates):
			if(recalculateAtraceUnoptimisedBio):
				_ , AprevLayer, _ = neuralNetworkPropagationCANNlayer(x, l-1, networkIndex, recordAtrace=False)
			else:
				AprevLayer = Atrace[generateParameterNameNetwork(networkIndex, l-1, "Atrace")] 
		else:
			AprevLayer = Aideal[generateParameterNameNetwork(networkIndex, l-1, "Aideal")]

		if(recalculateAtraceUnoptimisedBio):
			AtrialBase, ZtrialBase = neuralNetworkPropagationCANNlayerL(AprevLayer, l, networkIndex)
		else:
			AtrialBase = Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")]

		AidealLayer = Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")]

		if(useWeightUpdateDirectionHeuristicBasedOnExcitatoryInhibitorySynapseType):
			errorVec = calculateErrorAtrialVectorDirectional(AtrialBase, AidealLayer, networkIndex)
			#print("AidealDeltaOrigVec = ", AidealDeltaOrigVec)
			updateWeightsBasedOnAidealHeuristic(l, networkIndex, errorVec)
		else:
			lossBase = calculateErrorAtrial(AtrialBase, AidealLayer, networkIndex)
			updateWeightsBasedOnAidealStocastic(l, AprevLayer, AidealLayer, networkIndex, lossBase, x)

def updateWeightsBasedOnAidealHeuristic(l, networkIndex, errorVec):

	Wlayer = W[generateParameterNameNetwork(networkIndex, l, "W")]

	AidealDeltaTensorSizeW = tf.expand_dims(errorVec, axis=0)
	multiples = tf.constant([n_h[l-1],1], tf.int32)
	AidealDeltaOrigTensorSizeW = tf.tile(AidealDeltaTensorSizeW, multiples)

	AidealDeltaTensorSizeWSign = tf.sign(AidealDeltaTensorSizeW)
	
	learningRateW = learningRate	#useMultiplicationRatherThanAdditionOfDeltaValuesW: note effective weight learning rate is currently ~topLayerIdealAproximity*subLayerIdealAlearningRateBase*learningRate
	if(learningAlgorithm == "backpropApproximation5"):
		Wdelta = tf.multiply(AidealDeltaTensorSizeWSign, learningRateW)
	elif(learningAlgorithm == "backpropApproximation4"):
		if(useMultiplicationRatherThanAdditionOfDeltaValuesW):
			Wdelta = calculateDeltaTF(AidealDeltaTensorSizeW, learningRateW, useMultiplicationRatherThanAdditionOfDeltaValuesW)
		else:
			Wdelta = tf.multiply(AidealDeltaTensorSizeWSign, learningRateW)
	WlayerNew = tf.add(Wlayer, Wdelta)
	
	W[generateParameterNameNetwork(networkIndex, l, "W")] = WlayerNew

def updateWeightsBasedOnAidealStocastic(l, AprevLayer, AidealLayer, networkIndex, lossBase, x):

	#stocastic algorithm extracted from neuralNetworkPropagationCANN_expSUANNtrain_updateNeurons()g

	if(useBinaryWeights):
		variationDirections = 1
	else:
		variationDirections = 2
	
	for hIndexCurrentLayer in range(0, n_h[l]):	#k of l
		for hIndexPreviousLayer in range(0, n_h[l-1]+1):	#k of l-1
			if(hIndexPreviousLayer == n_h[l-1]):	#ensure that B parameter updates occur/tested less frequently than W parameter updates
				parameterTypeWorB = 0
			else:
				parameterTypeWorB = 1
			for variationDirectionInt in range(variationDirections):

				networkParameterIndexBase = (parameterTypeWorB, l, hIndexCurrentLayer, hIndexPreviousLayer, variationDirectionInt)
				networkParameterIndex = networkParameterIndexBase
	
				accuracyImprovementDetected = False
					
				if(not useBinaryWeights):
					if(networkParameterIndex[NETWORK_PARAM_INDEX_VARIATION_DIRECTION] == 1):
						variationDiff = learningRate
					else:
						variationDiff = -learningRate		
				
				if(networkParameterIndex[NETWORK_PARAM_INDEX_TYPE] == 1):
					currentVal = W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")][networkParameterIndex[NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER], networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].numpy()

					if(useBinaryWeights):
						if(useBinaryWeightsReduceMemoryWithBool):
							newVal = not currentVal
						else:
							newVal = float(not bool(currentVal))
					else:
						WtrialDelta = calculateDeltaNP(currentVal, variationDiff, useMultiplicationRatherThanAdditionOfDeltaValuesW)
						newVal = currentVal + WtrialDelta
						
					W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")][networkParameterIndex[NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER], networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].assign(newVal)
				else:
					currentVal = B[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "B")][networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].numpy()

					if(useBinaryWeights):
						if(useBinaryWeightsReduceMemoryWithBool):
							newVal = not currentVal
						else:
							newVal = float(not bool(currentVal))
					else:
						BtrialDelta = calculateDeltaNP(currentVal, variationDiff, useMultiplicationRatherThanAdditionOfDeltaValuesW)
						newVal = currentVal + BtrialDelta
					B[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "B")][networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].assign(newVal)
			
				Atrial, Ztrial = neuralNetworkPropagationCANNlayerL(AprevLayer, l, networkIndex)
				error = calculateErrorAtrial(Atrial, AidealLayer, networkIndex)	#average across all batches, across k neurons on l
				
				if(error < lossBase):
					accuracyImprovementDetected = True
					lossBase = error
					#print("\t(error < lossBase): error = ", error)				
							
				if(accuracyImprovementDetected):
					#print("accuracyImprovementDetected")
					Wbackup[generateParameterNameNetwork(networkIndex, l, "W")].assign(W[generateParameterNameNetwork(networkIndex, l, "W")])
					Bbackup[generateParameterNameNetwork(networkIndex, l, "B")].assign(B[generateParameterNameNetwork(networkIndex, l, "B")])								
				else:
					#print("!accuracyImprovementDetected")
					W[generateParameterNameNetwork(networkIndex, l, "W")].assign(Wbackup[generateParameterNameNetwork(networkIndex, l, "W")])
					B[generateParameterNameNetwork(networkIndex, l, "B")].assign(Bbackup[generateParameterNameNetwork(networkIndex, l, "B")])					



def activationFunction(Z, prevLayerSize=None):
	
	if(useBinaryWeights):	
		#offset required because negative weights are not used:
		Zoffset = tf.ones(Z.shape)
		Zoffset = tf.multiply(Zoffset, averageTotalInput)
		Zoffset = tf.multiply(Zoffset, prevLayerSize/2)
		Z = tf.subtract(Z, Zoffset) 
	
	if(activationFunctionType == "relu"):
		A = tf.nn.relu(Z)
	elif(activationFunctionType == "sigmoid"):
		A = tf.nn.sigmoid(Z)
	elif(activationFunctionType == "softmax"):
		A = tf.nn.softmax(Z)
		
	return A

 
def calculateDeltaTF(deltaMax, learningRateLocal, useMultiplication, applyMinimia=True):
	if(useMultiplication):
		deltaMaxAbs = tf.abs(deltaMax)
		deltaAbs = tf.multiply(deltaMaxAbs, learningRateLocal)
		if(applyMinimia):
			learningRateLocalMin = learningRateLocal*learningRateMinFraction
			deltaMinAbs = learningRateLocalMin
			deltaAbs = tf.maximum(deltaAbs, deltaMinAbs)
		deltaAbs = tf.minimum(deltaAbs, deltaMaxAbs)
		delta = tf.multiply(deltaAbs, tf.sign(deltaMax))
	else:
		delta = learningRateLocal	#tf.multiply(deltaMax, learningRateLocal)
	return delta

def calculateDeltaNP(deltaMax, learningRateLocal, useMultiplication, applyMinimia=True):
	if(useMultiplication):
		deltaMaxAbs = np.abs(deltaMax)
		deltaAbs = np.multiply(deltaMaxAbs, learningRateLocal)
		if(applyMinimia):
			learningRateLocalMin = learningRateLocal*learningRateMinFraction
			deltaMinAbs = learningRateLocalMin
			deltaAbs = np.maximum(deltaAbs, deltaMinAbs)
		deltaAbs = np.minimum(deltaAbs, deltaMaxAbs)
		delta = np.multiply(deltaAbs, np.sign(deltaMax))
	else:
		delta = learningRateLocal	#np.multiply(deltaMax, learningRateLocal) 
	return delta
		
def calculateErrorAtrialVectorDirectional(Atrial, AidealLayer, networkIndex, averageType="vector"):
	
	errorVec = calculateErrorAtrial(Atrial, AidealLayer, networkIndex, averageType)

	return errorVec
		
def calculateErrorAtrial(Atrial, AidealLayer, networkIndex, averageType="all"):

	AidealDelta = calculateADelta(AidealLayer, Atrial)
	error = AidealDelta

	if(averageAerrorAcrossBatch):
		if(averageType == "all"):
			error = tf.reduce_mean(error)	#average across k neurons on l
		elif(averageType == "vector"):
			error = error
		elif(averageType == "none"):
			error = error
	else:
		if(averageType == "all"):
			error = tf.reduce_mean(error)	#average across all batches, across k neurons on l
		elif(averageType == "vector"):
			error = tf.reduce_mean(error, axis=0)	#average across all batches	
		elif(averageType == "none"):
			error = error
									
	return error	
				
		
def modifyTensorRowColumn(a, isRow, index, updated_value, isVector):
	
	if(not isRow):
		a = tf.transpose(a)
		if(isVector):
			updated_value = tf.transpose(updated_value)
	
	if(index == 0):
		if(isVector):
			values = [updated_value, a[index+1:]]
		else:
			values = [[updated_value], a[index+1:]]
	elif(index == a.shape[0]-1):
		if(isVector):
			values = [a[:index], updated_value]
		else:
			values = [a[:index], [updated_value]]
	else:
		if(isVector):
			values = [a[:index], updated_value, a[index+1:]]
		else:
			values = [a[:index], [updated_value], a[index+1:]]
			
	a = tf.concat(axis=0, values=values)
			
	if(not isRow):
		a = tf.transpose(a)
		
	return a

def calculateAidealDelta(A, l, networkIndex):
	AidealDelta = calculateADelta(Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")], A)
	return AidealDelta
	
def calculateADelta(Abase, A):
	AidealDelta =  tf.subtract(Abase, A)
	return AidealDelta
		
def setAerror(AerrorLayer, AtraceLayer, l, networkIndex=1):
	if(errorStorageAlgorithm == "useAideal"):
		Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")] = tf.add(AtraceLayer, AerrorLayer)
	elif(errorStorageAlgorithm == "useAerror"):
		#print("setAerror, AerrorLayer.shape = ", AerrorLayer.shape)
		Aerror[generateParameterNameNetwork(networkIndex, l, "Aerror")] = AerrorLayer

def setAerrorGivenAideal(AidealLayer, AtraceLayer, l, networkIndex=1):
	if(errorStorageAlgorithm == "useAideal"):
		Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")] = AidealLayer
	elif(errorStorageAlgorithm == "useAerror"):
		#print("setAerrorGivenAideal, AidealLayer.shape = ", AidealLayer.shape)
		#print("setAerrorGivenAideal, AtraceLayer.shape = ", AtraceLayer.shape)
		Aerror[generateParameterNameNetwork(networkIndex, l, "Aerror")] = tf.subtract(AidealLayer, AtraceLayer)

			


