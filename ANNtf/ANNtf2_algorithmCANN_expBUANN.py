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

- Author: Richard Bruce Baxter - Copyright (c) 2020 Baxter AI (baxterai.com)

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *
import ANNtf2_operations
import ANNtf2_globalDefs
import math
from numpy import random



debugOnlyTrainFinalLayer = True	#debug weight update method only (not Aideal calculation method)

if(debugOnlyTrainFinalLayer):
	debugOnlyTrainFinalLayerSoftmaxLoss = False	#simulate SUANN performance with debugOnlyTrainFinalLayer
useSoftMaxOutputLayerDuringTraining = False	#else use Relu to normalise the training process

debugVerboseOutput = False
debugVerboseOutputTrain = False

useWeightUpdateDirectionHeuristicBasedOnExcitatoryInhibitorySynapseType = False	#enables rapid weight updates, else use stocastic (test both +/-) weight upates

#activationFunctionType = "relu"
activationFunctionType = "sigmoid"

updateWeightsAfterAidealCalculations = True	#method 1
updateWeightsDuringAidealCalculations = False	#method 2
updateWeightsBeforeAidealCalculations = False	#method 3
takeAprevLayerFromTraceRecalculateBeforeWeightUpdates = False
if(updateWeightsAfterAidealCalculations):
	takeAprevLayerFromTrace = False
elif(updateWeightsDuringAidealCalculations):
	takeAprevLayerFromTrace = True
	if(takeAprevLayerFromTrace):
		takeAprevLayerFromTraceRecalculateBeforeWeightUpdates = True 
elif(updateWeightsBeforeAidealCalculations):
	takeAprevLayerFromTrace = True

learningRateMinFraction = 0.1	#minimum learning rate can be set to always be above 0 (learningRateMinFraction = fraction of learning rate)



useMultiplicationRatherThanAdditionOfDeltaValues = False	#this ensures that Aideal/weight updates are normalised across their local layer (to minimise the probability an alternate class data propagation will be interferred with by the update)
if(useMultiplicationRatherThanAdditionOfDeltaValues):
	useMultiplicationRatherThanAdditionOfDeltaValuesAideal = False
	useMultiplicationRatherThanAdditionOfDeltaValuesW = True
else:
	useMultiplicationRatherThanAdditionOfDeltaValuesAideal = False
	useMultiplicationRatherThanAdditionOfDeltaValuesW = False
	


#BUANN is currently implemented to calculate independent idealA for each batch index (rather than averaged across batch)

applyMinimiumAdeltaContributionThreshold = False 	#only adjust Aideal_k of l based on Aideal of l+1 if it significantly improves Aideal of l+1, where k is neuron index of l
if(applyMinimiumAdeltaContributionThreshold):
	minimiumAdeltaContributionThreshold = 0.1	#fraction relative to original performance difference
	#minimiumAdeltaContributionThreshold = 1.0	#this contribution threshold is normalised wrt number of neurons (k) on l+1. default=1.0: if a Aideal_k adjustment on l contributes less than what on average an Aideal_k adjustment must necessarily contribute to achieve Aideal on l+1, then do not adjust Aideal_k (leave same as A_k)

topLayerIdealAstrict = True #top level learning target (idealA) == y, else learning target (idealA) == A + deltaA
topLayerIdealAproximity = 0.01	#maximum learning rate (effective learning rate will be less than this)
applySubLayerIdealAmultiplierRequirement = True
if(applySubLayerIdealAmultiplierRequirement):
	subLayerIdealAmultiplierRequirement = 1.5 #idealA of each neuron k on l will only be adjusted if its modification achieves at least xM performance increase for Aideal on l+1
	applySubLayerIdealAmultiplierCorrection = True	#optional: adjust learning neuron learning based on performance multiplier
else:
	applySubLayerIdealAmultiplierCorrection = False

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

Atrace = {}
Aideal = {}

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
			numEpochs = 10
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
		
			Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.dtypes.float32))
			Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.dtypes.float32))

	

def neuralNetworkPropagationCANN(x, networkIndex=1, recordAtrace=False):
	pred, A, Z = neuralNetworkPropagationCANNlayer(x, lTrain=numberOfLayers, networkIndex=networkIndex)
	return pred
	
def neuralNetworkPropagationCANNlayer(x, lTrain, networkIndex=1, recordAtrace=False):
			
	AprevLayer = x
	Atrace[generateParameterNameNetwork(networkIndex, 0, "Atrace")] = AprevLayer	#CHECKTHIS
	
	for l in range(1, lTrain+1):
	
		if(debugVerboseOutput):
			print("l = " + str(l))
			print("W = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
		
		A, Z = neuralNetworkPropagationCANNlayerL(AprevLayer, l, networkIndex)

		Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")] = A
		
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
	costCrossEntropyWithLogits = False	#binary classification
	loss = ANNtf2_operations.crossEntropy(pred, y, datasetNumClasses, costCrossEntropyWithLogits=costCrossEntropyWithLogits)
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
	pred, A, Z = neuralNetworkPropagationCANNlayer(x, numberOfLayers, networkIndex, recordAtrace=True)
	if(useSoftMaxOutputLayerDuringTraining):
		y_pred = pred	#pred is after softmax	
	else:
		y_pred = A	#A is after relu
	
	#2. calculate Aideal;
	calculateAidealTopLayer(y_pred, y_true, networkIndex)			
	calculateAidealBottomLayer(x, minLayerToTrain, networkIndex)		
	
	#3. calculate W updates;
	if(updateWeightsAfterAidealCalculations):
		for l in reversed(range(minLayerToTrain, numberOfLayers)):
			if(debugVerboseOutputTrain):
				print("calculateAideal: l = ", l)
			calculateAideal(l, networkIndex)	
		for l in range(minLayerToTrain, numberOfLayers+1):	#optimisation note: this can be done in parallel (weights can be updated for each layer simultaneously)
			#print("updateWeightsBasedOnAideal: l = ", l)
			updateWeightsBasedOnAideal(l, x, y, networkIndex)
	elif(updateWeightsDuringAidealCalculations):
		for l in reversed(range(minLayerToTrain, numberOfLayers+1)):	#2: do not calculate Aideal for input layer as this is always set to x
			if(debugVerboseOutputTrain):
				print("calculateAideal: l = ", l)
			if(l != 1):
				calculateAideal(l-1, networkIndex)	
			updateWeightsBasedOnAideal(l, x, y, networkIndex)
	elif(updateWeightsBeforeAidealCalculations):
		for l in reversed(range(minLayerToTrain, numberOfLayers+1)):
			if(debugVerboseOutputTrain):
				print("calculateAideal: l = ", l)
			updateWeightsBasedOnAideal(l, x, y, networkIndex)
			if(l != 1):
				calculateAideal(l-1, networkIndex)


def calculateAidealTopLayer(y_pred, y_true, networkIndex=1):

	#calculate Aideal of final layer based on y	
	AdeltaMax = tf.subtract(y_true, y_pred)	

	if(topLayerIdealAstrict):
		Adelta = AdeltaMax
	else:
		#print("calculateAidealTopLayer warning: ")
		Adelta = calculateDeltaTF(AdeltaMax, topLayerIdealAproximity, True, applyMinimia=False)
		
	Aideal[generateParameterNameNetwork(networkIndex, numberOfLayers, "Aideal")] = tf.add(y_pred, Adelta)
	
def calculateAidealBottomLayer(x, minLayerToTrain, networkIndex=1):
	
	Aideal[generateParameterNameNetwork(networkIndex, 0, "Aideal")] = x	#set Aideal of input layer to x

	if(debugOnlyTrainFinalLayer):
		for l in range(1, minLayerToTrain):
			Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")] = Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")]	#set Aideal of input layer to Atrace


def calculateAideal(l, networkIndex=1):

	#stocastically identify Aideal of l (lower) based on Aideal of l+1
		#this is biologically achieved by temporarily/independently adjusting the firing rate (~bias) of each neuron (index k) on l, and seeing if this better achieves Aideal of l+1
		#feedback (positive/negtive trial) is given from higher level l+1 to l_k in the form of "simple" [boolean] ~local chemical signal

	A = Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")]	#get original A value of lower layer
	for k in range(n_h[l]):
		if(debugVerboseOutputTrain):
			print("\tcalculateAideal: k = ", k)
		#try both positive and negative adjustments of A_l_k;
		trialAidealMod(True, A, k, l, networkIndex)
		trialAidealMod(False, A, k, l, networkIndex)

def trialAidealMod(direction, A, k, l, networkIndex):

	if(direction):
		trialAmodValue = subLayerIdealAlearningRateBase
	else:
		trialAmodValue = -subLayerIdealAlearningRateBase
	
	columnsIdx = tf.constant([k])
	AK = tf.gather(A, columnsIdx, axis=1)	#Atrial[:,k]	
	AtrialK = AK
	AtrialKdelta = calculateDeltaTF(AtrialK, trialAmodValue, useMultiplicationRatherThanAdditionOfDeltaValuesAideal)
	AtrialK = tf.add(AtrialK, AtrialKdelta)
	
	Atrial = A
	Atrial = modifyTensorRowColumn(Atrial, False, k, AtrialK, isVector=True)	#Atrial[:,k] = (trialAmodValue)

	AtrialAbove, ZtrialAbove = neuralNetworkPropagationCANNlayerL(Atrial, l+1, networkIndex)
	successfulTrial, performanceMultiplier = testAtrialPerformance(AtrialKdelta, AtrialAbove, l, networkIndex)
	successfulTrialFloat = tf.dtypes.cast(successfulTrial, dtype=tf.float32)
	
	if(applySubLayerIdealAmultiplierCorrection):
		AtrialKdelta = tf.multiply(AtrialKdelta, performanceMultiplier)
			
	AtrialKdeltaSuccessful = tf.multiply(AtrialKdelta, successfulTrialFloat)
	
	AtrialKSuccessful = tf.add(AtrialK, AtrialKdeltaSuccessful)
	
	if(debugVerboseOutputTrain):
		print("AtrialKSuccessful", AtrialKSuccessful)
	
	Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")] = modifyTensorRowColumn(Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")], False, k, AtrialKSuccessful, isVector=True)
	
	return A


def testAtrialPerformance(AtrialKdelta, AtrialAbove, l, networkIndex):
		
	successfulTrial, trialPerformanceGain = testAtrialPerformanceAbove(AtrialAbove, l+1, networkIndex)
	
	successfulTrial = tf.expand_dims(successfulTrial, axis=1)
	trialPerformanceGain = tf.expand_dims(trialPerformanceGain, axis=1)
	
	performanceMultiplier = tf.divide(trialPerformanceGain, AtrialKdelta)	#TODO: fix this; sometimes divides by zero
		
	if(applySubLayerIdealAmultiplierRequirement):
		performanceMultiplierSuccessful = tf.greater(performanceMultiplier, subLayerIdealAmultiplierRequirement)
		successfulTrial = tf.logical_and(successfulTrial, performanceMultiplierSuccessful)
	
	return successfulTrial, performanceMultiplier
	

def testAtrialPerformanceAbove(AtrialAbove, l, networkIndex):
	
	AidealDeltaOrig = calculateAidealDelta(Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")], l, networkIndex)
	AidealDeltaTrial = calculateAidealDelta(AtrialAbove, l, networkIndex)
	AidealDeltaOrigAvg = tf.reduce_mean(AidealDeltaOrig, axis=1)   #average across all k neurons on l
	AidealDeltaTrialAvg = tf.reduce_mean(AidealDeltaTrial, axis=1) #average across all k neurons on l
	if(applyMinimiumAdeltaContributionThreshold):
		AidealDeltaOrigAvgThreshold = tf.multiply(tf.sign(AidealDeltaOrigAvg), tf.subtract(tf.abs(AidealDeltaOrigAvg), minimiumAdeltaContributionThreshold))
		#AidealDeltaOrigAvgThreshold = tf.multiply(tf.sign(AidealDeltaOrigAvg), tf.subtract(tf.abs(AidealDeltaOrigAvg), minimiumAdeltaContributionThreshold)) #tf.maximum(tf.subtract(tf.abs(AidealDeltaOrigAvg), minimiumAdeltaContributionThreshold), 0.0)
	else:
		AidealDeltaOrigAvgThreshold = AidealDeltaOrigAvg
		
	successfulTrial = tf.math.logical_not(tf.math.logical_xor(tf.less(AidealDeltaTrialAvg, AidealDeltaOrigAvgThreshold), tf.equal(tf.sign(AidealDeltaTrialAvg), 1)))    #tf.multiply(tf.less(AidealDeltaTrialAvg, AidealDeltaOrigAvg), tf.sign(AidealDeltaTrialAvg))
	
	trialPerformanceGain = tf.multiply(tf.subtract(AidealDeltaOrigAvg, AidealDeltaTrialAvg), tf.sign(AidealDeltaTrialAvg))	#orig trialPerformanceGain calculation method 
	
	return successfulTrial, trialPerformanceGain
	
def calculateAidealDelta(A, l, networkIndex):
	AidealDelta =  calculateADelta(Aideal[generateParameterNameNetwork(networkIndex, l, "Aideal")], A, l)
	return AidealDelta
	
def calculateADelta(Abase, A, l):
	AidealDelta =  tf.subtract(Abase, A)
	return AidealDelta
		
	


		
		
def updateWeightsBasedOnAideal(l, x, y, networkIndex):
	
	if(takeAprevLayerFromTrace):
		if(takeAprevLayerFromTraceRecalculateBeforeWeightUpdates):
			neuralNetworkPropagationCANNlayer(x, l-1, networkIndex, recordAtrace=True)
		AprevLayer = Atrace[generateParameterNameNetwork(networkIndex, l-1, "Atrace")] 
	else:
		AprevLayer = Aideal[generateParameterNameNetwork(networkIndex, l-1, "Aideal")]
		
	AtrialBase, ZtrialBase = neuralNetworkPropagationCANNlayerL(AprevLayer, l, networkIndex)

	AidealDeltaOrig = calculateAidealDelta(AtrialBase, l, networkIndex)
	
	if(useWeightUpdateDirectionHeuristicBasedOnExcitatoryInhibitorySynapseType):
		AidealDeltaOrigVec = tf.reduce_mean(AidealDeltaOrig, axis=0)	#average across all batches
		#print("AidealDeltaOrigVec = ", AidealDeltaOrigVec)
		updateWeightsBasedOnAidealHeuristic(l, networkIndex, AidealDeltaOrigVec)
	else:
		lossBase = calculateLossAdelta(AidealDeltaOrig)
		if(debugOnlyTrainFinalLayerSoftmaxLoss):
			lossBase, accBase = neuralNetworkPropagationCANN_test(x, y, networkIndex)	#debug only
		updateWeightsBasedOnAidealStocastic(l, AprevLayer, networkIndex, lossBase, x, y)

def updateWeightsBasedOnAidealHeuristic(l, networkIndex, AidealDeltaVec):

	Wlayer = W[generateParameterNameNetwork(networkIndex, l, "W")]

	AidealDeltaTensorSizeW = tf.expand_dims(AidealDeltaVec, axis=0)
	multiples = tf.constant([n_h[l-1],1], tf.int32)
	AidealDeltaOrigTensorSizeW = tf.tile(AidealDeltaTensorSizeW, multiples)

	AidealDeltaTensorSizeWSign = tf.sign(AidealDeltaTensorSizeW)
	
	learningRateW = learningRate	#note effective weight learning rate is currently ~topLayerIdealAproximity*subLayerIdealAlearningRateBase*learningRate
	if(useMultiplicationRatherThanAdditionOfDeltaValuesW):
		Wdelta = calculateDeltaTF(AidealDeltaTensorSizeW, learningRateW, useMultiplicationRatherThanAdditionOfDeltaValuesW)
	else:
		Wdelta = tf.multiply(AidealDeltaTensorSizeWSign, learningRateW)
	WlayerNew = tf.add(Wlayer, Wdelta)
	
	W[generateParameterNameNetwork(networkIndex, l, "W")] = WlayerNew

def updateWeightsBasedOnAidealStocastic(l, AprevLayer, networkIndex, lossBase, x, y):

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

				AidealDeltaTrial = calculateAidealDelta(Atrial, l, networkIndex)	#ADeltaTrial = calculateADelta(Atrial, AtrialBase, l)
				loss = calculateLossAdelta(AidealDeltaTrial)	#average across all batches, across k neurons on l
				
				if(debugOnlyTrainFinalLayerSoftmaxLoss):
					pred = tf.nn.softmax(Ztrial)
					costCrossEntropyWithLogits = False	#binary classification
					loss = ANNtf2_operations.crossEntropy(pred, y, datasetNumClasses, costCrossEntropyWithLogits=costCrossEntropyWithLogits)
					#loss, acc = neuralNetworkPropagationCANN_test(x, y, networkIndex)	#method2
				
				if(loss < lossBase):
					accuracyImprovementDetected = True
					lossBase = loss
					#print("\t(loss < lossBase): loss = ", loss)				
							
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
	
	return A

 
def calculateDeltaTF(deltaMax, learningRateLocal, useMultiplication, applyMinimia=True):
	if(useMultiplication):
		deltaMaxAbs = tf.abs(deltaMax)
		deltaAbs = tf.multiply(deltaMaxAbs, learningRateLocal)
		if(applyMinimia):
			learningRateLocalMin = learningRateLocal*learningRateLocalMinFraction
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
			learningRateLocalMin = learningRateLocal*learningRateLocalMinFraction
			deltaMinAbs = learningRateLocalMin
			deltaAbs = np.maximum(deltaAbs, deltaMinAbs)
		deltaAbs = np.minimum(deltaAbs, deltaMaxAbs)
		delta = np.multiply(deltaAbs, np.sign(deltaMax))
	else:
		delta = learningRateLocal	#np.multiply(deltaMax, learningRateLocal) 
	return delta
		

def calculateLossAdelta(AidealDeltaOrig):
	AidealDeltaOrig = tf.abs(AidealDeltaOrig)
	AidealDeltaOrigAvg = tf.reduce_mean(AidealDeltaOrig)	#average across all batches, across k neurons on l
	return AidealDeltaOrigAvg
		
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



