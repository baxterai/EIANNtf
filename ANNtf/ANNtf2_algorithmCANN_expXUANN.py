# -*- coding: utf-8 -*-
"""ANNtf2_algorithmCANN_expXUANN.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Usage:
see ANNtf2.py

# Description

Define fully connected contrastive (pos/neg sample diff) update artificial neural network (CANN_expXUANN)

- Author: Richard Bruce Baxter - Copyright (c) 2020 Baxter AI (baxterai.com)

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *
import ANNtf2_operations
import ANNtf2_globalDefs
import math
from numpy import random

learningRate = 0.001

debugWexplosion = False
debugFastTrain = True

objectiveTargetMinimiseDiffBetweenPositiveSamples = True

#learning algorithm embedded in forward propagation of new class x experience following forward propagation of existing class x experience
useBatch = True
if(useBatch):
	batchSize = 100
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
#Atrace = [{}]
numberOfTraces = 3	#current, positive sample, negative sample
traceIndexCurrent = 0
traceIndexPositiveSample = 1
traceIndexNegativeSample = 2


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
				trainingSteps = batchSize
			else:
				trainingSteps = 1000
		numEpochs = 10
	else:
		if(dataset == "POStagSequence"):
			trainingSteps = 10000
		elif(dataset == "SmallDataset"):
			if(debugFastTrain):
				trainingSteps = batchSize
			else:
				trainingSteps = 1000
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
			
			#for t in range(numberOfTraces):
			#	Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")][t] = tf.Variable(tf.zeros(n_h[l], dtype=tf.dtypes.float32))

	

def neuralNetworkPropagationCANNfinal(x, networkIndex=1):
	return neuralNetworkPropagationCANN(x, networkIndex, enableFinalLayerWeightUpdatesOnly=True)

def neuralNetworkPropagationCANN(x, networkIndex=1, enableFinalLayerWeightUpdatesOnly=False):
	
	global averageTotalInput
		
	AprevLayer = x

	#if(useBinaryWeights):
	#	if(averageTotalInput == -1):
	#		averageTotalInput = tf.math.reduce_mean(x)
	#		print("averageTotalInput = ", averageTotalInput)	 
	#print("x = ", x)
	
	for l in range(1, numberOfLayers+1):
	
		#print("l = " + str(l))
		
		if(useBinaryWeights):
			if(useBinaryWeightsReduceMemoryWithBool):
				Wfloat = tf.dtypes.cast(W[generateParameterNameNetwork(networkIndex, l, "W")], dtype=tf.float32)
				Bfloat = tf.dtypes.cast(B[generateParameterNameNetwork(networkIndex, l, "B")], dtype=tf.float32)
				Z = tf.add(tf.matmul(AprevLayer, Wfloat), Bfloat)
			else:
				Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
			A = reluCustom(Z, n_h[l-1])
		else:
			Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
			A = reluCustom(Z)
			
		AprevLayer = A
		
		if(enableFinalLayerWeightUpdatesOnly):
			if(l < numberOfLayers-1):
				AprevLayer = tf.stop_gradient(AprevLayer)
		
	pred = tf.nn.softmax(Z)
	
	#print("neuralNetworkPropagationCANN pred.shape = ", pred.shape)	

	return pred

def neuralNetworkPropagationCANNsub(x, samplePositiveX, sampleNegativeX, lTrain, networkIndex=1):
	
	AprevLayer = [x, samplePositiveX, sampleNegativeX]

	#if(useBinaryWeights):
	#	if(averageTotalInput == -1):
	#		averageTotalInput = tf.math.reduce_mean(x)
	#		print("averageTotalInput = ", averageTotalInput)	 
	
	#print("x.shape = ", x.shape)
	#print("samplePositiveX.shape = ", samplePositiveX.shape)
	#print("sampleNegativeX.shape = ", sampleNegativeX.shape)
	
	for l in range(1, lTrain+1):

		#print("l = " + str(l))
		
		#propagate each activation trace independently (x, samplePositiveX, sampleNegativeX);
		for t in range(numberOfTraces):
	
			#print("t = " + str(t))

			if(useBinaryWeights):
				if(useBinaryWeightsReduceMemoryWithBool):
					Wfloat = tf.dtypes.cast(W[generateParameterNameNetwork(networkIndex, l, "W")], dtype=tf.float32)
					Bfloat = tf.dtypes.cast(B[generateParameterNameNetwork(networkIndex, l, "B")], dtype=tf.float32)
					Z = tf.add(tf.matmul(AprevLayer[t], Wfloat), Bfloat)
				else:
					Z = tf.add(tf.matmul(AprevLayer[t], W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
				A = reluCustom(Z, n_h[l-1])
			else:
				Z = tf.add(tf.matmul(AprevLayer[t], W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
				A = reluCustom(Z)
	
			if(l < lTrain-1):
				AprevLayer[t] = tf.stop_gradient(AprevLayer[t])
					
			AprevLayer[t] = A
							
		if(l == lTrain):
			#add additional intermediary artificial output layer (for contrastive boolean objective function)
			
			#perform activation contrast
			#print("batchSize = ", batchSize)
			#print("AprevLayer[traceIndexCurrent].shape = ", AprevLayer[traceIndexCurrent].shape)
			positiveDiff = tf.subtract(AprevLayer[traceIndexCurrent], AprevLayer[traceIndexPositiveSample])
			negativeDiff = tf.subtract(AprevLayer[traceIndexCurrent], AprevLayer[traceIndexNegativeSample])
			#print("negativeDiff.shape = ", negativeDiff.shape)
			positiveDiffavg = tf.math.reduce_mean(positiveDiff, axis=1)
			negativeDiffavg = tf.math.reduce_mean(negativeDiff, axis=1)
			#print("positiveDiffavg.shape = ", positiveDiffavg.shape)
			ZlastLayer = tf.concat([positiveDiffavg, negativeDiffavg], 0)
			#print("ZlastLayer.shape = ", ZlastLayer.shape)
			
			#pred = tf.nn.softmax(ZlastLayer)
			pred = tf.nn.sigmoid(ZlastLayer)	#binary classification	
			
			#print("neuralNetworkPropagationCANNsub: pred.shape = ", pred.shape)	

	#print("neuralNetworkPropagationCANN pred.shape = ", pred.shape)
	
	return pred
	
		


def neuralNetworkPropagationCANN_test(x, y, networkIndex=1):

	pred = neuralNetworkPropagationCANN(x, networkIndex)
	costCrossEntropyWithLogits = False	#binary classification
	loss = ANNtf2_operations.crossEntropy(pred, y, datasetNumClasses, costCrossEntropyWithLogits=costCrossEntropyWithLogits)
	acc = ANNtf2_operations.calculateAccuracy(pred, y)
	
	return loss, acc
	
	
def neuralNetworkPropagationCANN_expXUANNtrain(x, y, samplePositiveX, samplePositiveY, sampleNegativeX, sampleNegativeY, networkIndex=1):
	for l in range(1, numberOfLayers):
		executeOptimisationSub(x, y, samplePositiveX, samplePositiveY, sampleNegativeX, sampleNegativeY, l, networkIndex)
	executeOptimisationFinal(x, y, networkIndex)

def executeOptimisationSub(x, y, samplePositiveX, samplePositiveY, sampleNegativeX, sampleNegativeY, lTrain, networkIndex=1):

	if(objectiveTargetMinimiseDiffBetweenPositiveSamples):
		yIntermediaryArtificialTargetPart1 = tf.zeros(samplePositiveY.shape, dtype=tf.dtypes.float32)
		yIntermediaryArtificialTargetPart2 = tf.ones(sampleNegativeY.shape, dtype=tf.dtypes.float32)
	else:
		yIntermediaryArtificialTargetPart1 = tf.ones(samplePositiveY.shape, dtype=tf.dtypes.float32)
		yIntermediaryArtificialTargetPart2 = tf.zeros(sampleNegativeY.shape, dtype=tf.dtypes.float32)	
	yIntermediaryArtificialTarget = tf.concat([yIntermediaryArtificialTargetPart1, yIntermediaryArtificialTargetPart2], 0)
	yIntermediaryArtificialTargetNumClasses = y.shape[0]*2	#batchSize*2
	
	#debug sanity checks:
	#print("executeOptimisation: yIntermediaryArtificialTarget.shape = ", yIntermediaryArtificialTarget.shape)
	#print("executeOptimisation: batchSize*2 = ", batchSize*2)
	
	with tf.GradientTape() as g:
		pred = neuralNetworkPropagationCANNsub(x, samplePositiveX, sampleNegativeX, lTrain, networkIndex)
		#print("pred.shape = ", pred.shape)
		#print("yIntermediaryArtificialTarget.shape = ", yIntermediaryArtificialTarget.shape)
		loss = crossEntropy(pred, yIntermediaryArtificialTarget, yIntermediaryArtificialTargetNumClasses, costCrossEntropyWithLogits=True)	#single intermediary (per layer) output neuron used for training
		
	Wlist = []
	Blist = []
	#for l in range(1, lTrain+1):
	l = lTrain
	Wlist.append(W[generateParameterNameNetwork(networkIndex, l, "W")])
	Blist.append(B[generateParameterNameNetwork(networkIndex, l, "B")])
	
	trainableVariables = Wlist + Blist
	
	gradients = g.gradient(loss, trainableVariables)
	optimizer.apply_gradients(zip(gradients, trainableVariables))
	
def executeOptimisationFinal(x, y, networkIndex=1):

	with tf.GradientTape() as g:
		pred = neuralNetworkPropagationCANNfinal(x, networkIndex)
		loss = crossEntropy(pred, y, datasetNumClasses, costCrossEntropyWithLogits=False)
		
	Wlist = []
	Blist = []
	#for l in range(1, numberOfLayers+1):
	l = numberOfLayers
	Wlist.append(W[generateParameterNameNetwork(networkIndex, l, "W")])
	Blist.append(B[generateParameterNameNetwork(networkIndex, l, "B")])
	
	trainableVariables = Wlist + Blist
	
	gradients = g.gradient(loss, trainableVariables)
	optimizer.apply_gradients(zip(gradients, trainableVariables))
	
	
	
def reluCustom(Z, prevLayerSize=None):
	
	if(useBinaryWeights):	
		#offset required because negative weights are not used:
		Zoffset = tf.ones(Z.shape)
		Zoffset = tf.multiply(Zoffset, averageTotalInput)
		Zoffset = tf.multiply(Zoffset, prevLayerSize/2)
		#print("Zoffset = ", Zoffset)
		Z = tf.subtract(Z, Zoffset) 
		A = tf.nn.relu(Z)
		#AaboveZero = tf.math.greater(A, 0)
		#AaboveZeroFloat = tf.dtypes.cast(AaboveZero, dtype=tf.dtypes.float32)
		#ZoffsetRestore = AaboveZeroFloat*Zoffset
		#print("ZoffsetRestore = ", ZoffsetRestore)
		#A = tf.add(A, ZoffsetRestore)
	else:
		A = tf.nn.relu(Z)
	
	#print("Z = ", Z)
	#print("A = ", A)
	
	return A

 


def generateTFtrainDataFromNParraysCANN_expXUANN(train_x, train_y, shuffleSize, batchSize, datasetNumClasses):

	trainDataList = []
	
	for classTarget in range(datasetNumClasses):
			
		train_xClassFiltered, train_yClassFiltered = filterNParraysByClassTarget(train_x, train_y, classTargetFilterIndex=classTarget)
		trainDataUnbatched = generateTFtrainDataUnbatchedFromNParrays(train_xClassFiltered, train_yClassFiltered)
		trainData = generateTFtrainDataFromTrainDataUnbatched(trainDataUnbatched, shuffleSize, batchSize)
		trainDataList.append(trainData)
		
	return trainDataList
		
		




#def neuralNetworkPropagationCANN-legacy(x, networkIndex=1, recordAtrace=False, traceIndex=-1):
#	
#	global averageTotalInput
#		
#	AprevLayer = x
#
#	if(useBinaryWeights):
#		if(averageTotalInput == -1):
#			averageTotalInput = tf.math.reduce_mean(x)
#			print("averageTotalInput = ", averageTotalInput)
#			 
#	#print("x = ", x)
#	
#	for l in range(1, numberOfLayers+1):
#	
#		#print("l = " + str(l))
#		
#		if(useBinaryWeights):
#			if(useBinaryWeightsReduceMemoryWithBool):
#				Wfloat = tf.dtypes.cast(W[generateParameterNameNetwork(networkIndex, l, "W")], dtype=tf.float32)
#				Bfloat = tf.dtypes.cast(B[generateParameterNameNetwork(networkIndex, l, "B")], dtype=tf.float32)
#				Z = tf.add(tf.matmul(AprevLayer, Wfloat), Bfloat)
#			else:
#				Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
#			A = reluCustom(Z, n_h[l-1])
#		else:
#			Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
#			A = reluCustom(Z)
#			
#		if(recordAtrace):
#		   if(onlyTrainNeuronsIfActivationContributionAboveThreshold1):
#			   #apply threshold to A
#			   AAboveThreshold = tf.math.greater(A, onlyTrainNeuronsIfActivationContributionAboveThreshold1Value)
#			   AAboveThresholdFloat = tf.dtypes.cast(AAboveThreshold, dtype=tf.float32)
#			   ALearn = A*AAboveThresholdFloat
#		   else:
#			   ALearn = A
#		   #print("ALearn.shape = ", ALearn.shape)
#		   Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")][traceIndex] = ALearn
#			
#		AprevLayer = A
#		
#	pred = tf.nn.softmax(Z)
#	
#	#print("neuralNetworkPropagationCANN pred.shape = ", pred.shape)	
#
#	return pred

#def neuralNetworkPropagationCANN_expXUANNtrain-legacy(x, y, samplePositiveX, samplePositiveY, sampleNegativeX, sampleNegativeY, networkIndex=1):	#currentClassTarget
#	
#	#debug:
#	#print("batchSize = ", batchSize)
#	#print("learningRate = ", learningRate)
#	#print("x = ", x)
#			
#	predExemplars = neuralNetworkPropagationCANN_expXUANN(exemplarsX, networkIndex, recordAtrace=True, traceIndex=0)
#	predExemplars = neuralNetworkPropagationCANN_expXUANN(samplePositiveX, networkIndex, recordAtrace=True, traceIndex=1)
#	predExemplars = neuralNetworkPropagationCANN_expXUANN(sampleNegativeX, networkIndex, recordAtrace=True, traceIndex=2)
#	
#	AprevLayer = x
#
#	for l in range(1, numberOfLayers+1):
#
#		if(debugWexplosion):
#			print("l = " + str(l))
#			print("W[generateParameterNameNetwork(networkIndex, l, W)] = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
#		
#		
#		
#	#clear Atrace;
#	for l in range(1, numberOfLayers+1):
#		for t in range(numberOfTraces):
#			Atrace[generateParameterNameNetwork(networkIndex, l, "Atrace")][traceIndex] = 0	#tf.zeros(n_h[l])
#		
#	pred = neuralNetworkPropagationCANN_expXUANN(x, networkIndex)
#	
#	return pred
#
	
	
	
