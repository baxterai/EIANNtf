# -*- coding: utf-8 -*-
"""SANItf2_algorithmCANN.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Usage:
see SANItf2.py

# Description

Define fully connected stochastically updated artificial neural network (SUANN)

- Author: Richard Bruce Baxter - Copyright (c) 2020 Baxter AI (baxterai.com)

"""

import tensorflow as tf
import numpy as np
from SANItf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import SANItf2_operations
#from SANItf2_operations import generateParameterNameNetwork
import SANItf2_globalDefs
import math
from numpy import random

maximumNetworkHiddenLayerNeuronsAsFractionOfInputNeurons = 0.8	#0.8	#10

numberOfSubsetsTrialledPerBaseParameter = 10	#will affect speed

generateNetworkNonlinearConvergence = False
if(generateNetworkNonlinearConvergence):
	networkDivergenceType = "nonLinearConverging"
	if(applyNeuronThresholdBias):
		#this will affect the optimimum convergence angle
		networkOptimumConvergenceAngle = 0.5+applyNeuronThresholdBiasValue
	else:
		networkOptimumConvergenceAngle = 0.5	#if angle > 0.5, then more obtuse triange, if < 0.5 then more acute triangle	#fractional angle between 0 and 90 degrees
	networkDivergence = 1.0-networkOptimumConvergenceAngle 
	#required for Logarithms with a Fraction as Base:
	networkDivergenceNumerator = int(networkDivergence*10)
	networkDivergenceDenominator = 10
else:
	networkDivergenceType = "linearConverging"
	#networkDivergenceType = "linearDivergingThenConverging"	#not yet coded
	

W = {}
B = {}

Wbackup = {}
Bbackup = {}

NETWORK_PARAM_INDEX_TYPE = 0
NETWORK_PARAM_INDEX_LAYER = 1
NETWORK_PARAM_INDEX_H_CURRENT_LAYER = 2
NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER = 3
NETWORK_PARAM_INDEX_VARIATION_DIRECTION = 4


#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0
datasetNumClasses = 0

learningRate = 0.0
batchSize = 0

#randomNormal = tf.initializers.RandomNormal()

def defineTrainingParametersSUANN(dataset, trainMultipleFiles):

	global learningRate
	global forgetRate
	global batchSize
	
	if(trainMultipleFiles):
		learningRate = 0.0001
		if(dataset == "POStagSequence"):
			trainingSteps = 10000
		elif(dataset == "NewThyroid"):
			trainingSteps = 1000
		batchSize = 100
	else:
		learningRate = 0.001
		if(dataset == "POStagSequence"):
			trainingSteps = 10000
		elif(dataset == "NewThyroid"):
			trainingSteps = 1000
		batchSize = 10	#1	#10	#1000	#temporarily reduce batch size for visual debugging (array length) purposes)
	displayStep = 100
	
	return learningRate, trainingSteps, batchSize, displayStep
	

def defineNetworkParametersSUANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	global datasetNumClasses
	
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
		elif(dataset == "NewThyroid"):
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
			print("defineNetworkParametersSUANN error: linearDivergingThenConverging not yet coded")
			exit()
		else:
			print("defineNetworkParametersSUANN error: unknown networkDivergenceType")
			exit()

	n_h_last = n_y
	n_h.append(n_h_last)
	
	print("defineNetworkParametersSUANN, n_h = ", n_h)

	return numberOfLayers
	
	
def neuralNetworkPropagationSUANN(x, networkIndex=1):
			
	AprevLayer = x
	 
	for l in range(1, numberOfLayers+1):
	
		#print("l = " + str(l))

		Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
		A = relu(Z)
		#A = tf.nn.relu(Z)
			
		AprevLayer = A
			
	return tf.nn.softmax(Z)
	

def neuralNetworkPropagationSUANNtest(x, y, networkIndex=1):

	pred = neuralNetworkPropagationSUANN(x, networkIndex)
	loss = SANItf2_operations.crossEntropy(pred, y, datasetNumClasses, costCrossEntropyWithLogits=False)
	acc = SANItf2_operations.calculateAccuracy(pred, y)
	
	return loss, acc
	
def neuralNetworkPropagationSUANNtrain(x, y=None, networkIndex=1, parameterUpdateSubsetSize=1):

	#print("batchSize = ", batchSize)
	#print("learningRate = ", learningRate)
	
	#print("x = ", x)
	
	lossStart, accStart = neuralNetworkPropagationSUANNtest(x, y, networkIndex)	#debug only
	print("lossStart = ", lossStart)
	
	#ensure that an update is tried at least once for each parameter of the network during each training iteration:
		
	for l in range(1, numberOfLayers+1):
		
		#print("l = ", l)
		#print("W = ", W[generateParameterNameNetwork(networkIndex, l, "W")])
		#print("B = ", B[generateParameterNameNetwork(networkIndex, l, "B")])
			
		for hIndexCurrentLayer in range(0, n_h[l]):
			for hIndexPreviousLayer in range(0, n_h[l-1]+1):
				if(hIndexPreviousLayer == n_h[l-1]):	#ensure that B parameter updates occur/tested less frequently than W parameter updates
					parameterTypeWorB = 0
				else:
					parameterTypeWorB = 1
				for variationDirectionInt in range(2):
	
					networkParameterIndexBase = (parameterTypeWorB, l, hIndexCurrentLayer, hIndexPreviousLayer, variationDirectionInt)
			
					lossBase, accBase = neuralNetworkPropagationSUANNtest(x, y, networkIndex)
					
					#print("hIndexCurrentLayer = ", hIndexCurrentLayer)
					#print("hIndexPreviousLayer = ", hIndexPreviousLayer)
					#print("parameterTypeWorB = ", parameterTypeWorB)
					#print("variationDirectionInt = ", variationDirectionInt)
					
					for subsetTrialIndex in range(0, numberOfSubsetsTrialledPerBaseParameter):

						accuracyImprovementDetected = False

						currentSubsetOfParameters = []
						currentSubsetOfParameters.append(networkParameterIndexBase)

						for s in range(1, parameterUpdateSubsetSize):
							networkParameterIndex = getRandomNetworkParameter(networkIndex, currentSubsetOfParameters)
							currentSubsetOfParameters.append(networkParameterIndex)

						for s in range(0, parameterUpdateSubsetSize):
							networkParameterIndex = currentSubsetOfParameters[s]
							
							if(networkParameterIndex[NETWORK_PARAM_INDEX_VARIATION_DIRECTION] == 1):
								variationDiff = learningRate
							else:
								variationDiff = -learningRate							
														
							if(networkParameterIndex[NETWORK_PARAM_INDEX_TYPE] == 1):
								Wnp = W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")].numpy()
								currentVal = Wnp[networkParameterIndex[NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER], networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]]
								#print("currentVal = ", currentVal)
								#print("W1 = ", W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")])
								
								W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")][networkParameterIndex[NETWORK_PARAM_INDEX_H_PREVIOUS_LAYER], networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].assign(currentVal + variationDiff)
						
								#print("W2 = ", W[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "W")])
							else:
								Bnp = B[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "B")].numpy()
								currentVal = Bnp[networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]]
								B[generateParameterNameNetwork(networkIndex, networkParameterIndex[NETWORK_PARAM_INDEX_LAYER], "B")][networkParameterIndex[NETWORK_PARAM_INDEX_H_CURRENT_LAYER]].assign(currentVal + variationDiff)
			
						loss, acc = neuralNetworkPropagationSUANNtest(x, y, networkIndex)
						#print("loss = ", loss)
						
						if(loss < lossBase):
							accuracyImprovementDetected = True
							lossBase = loss
							#print("(loss < lossBase): loss = ", loss)						
						
						if accuracyImprovementDetected:
							#print("accuracyImprovementDetected")
							Wbackup[generateParameterNameNetwork(networkIndex, l, "W")].assign(W[generateParameterNameNetwork(networkIndex, l, "W")])
							Bbackup[generateParameterNameNetwork(networkIndex, l, "B")].assign(B[generateParameterNameNetwork(networkIndex, l, "B")])								
						else:
							#print("!accuracyImprovementDetected")
							W[generateParameterNameNetwork(networkIndex, l, "W")].assign(Wbackup[generateParameterNameNetwork(networkIndex, l, "W")])
							B[generateParameterNameNetwork(networkIndex, l, "B")].assign(Bbackup[generateParameterNameNetwork(networkIndex, l, "B")])					
		
	pred = neuralNetworkPropagationSUANN(x, networkIndex)
	
	return pred
									

def getRandomNetworkParameter(networkIndex, currentSubsetOfParameters):
	
	foundNewParameter = False
	while not foundNewParameter:
	
		variationDirection = random.randint(2)
		layer = random.randint(1, len(n_h))
		parameterTypeWorBtemp = random.randint(n_h[layer-1]+1)	#ensure that B parameter updates occur/tested less frequently than W parameter updates	#OLD: random.randint(2)	
		if(parameterTypeWorBtemp == n_h[layer-1]):
			parameterTypeWorB = 0
		else:
			parameterTypeWorB = 1
		hIndexCurrentLayer = random.randint(n_h[layer])	#randomNormal(n_h[l])
		hIndexPreviousLayer = random.randint(n_h[layer-1]) #randomNormal(n_h[l-1])
		networkParameterIndex = (parameterTypeWorB, layer, hIndexCurrentLayer, hIndexPreviousLayer, variationDirection)
	
		matches = [item for item in currentSubsetOfParameters if item == networkParameterIndex]
		if len(matches) == 0:
			foundNewParameter = True
			
	return networkParameterIndex

	

def defineNeuralNetworkParametersSUANN():

	randomNormal = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):
	
		for l in range(1, numberOfLayers+1):

			W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(randomNormal([n_h[l-1], n_h[l]]))
			B[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(tf.zeros(n_h[l]))
			
			Wbackup[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(W[generateParameterNameNetwork(networkIndex, l, "W")])
			Bbackup[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(B[generateParameterNameNetwork(networkIndex, l, "B")])
	
			#print(W[generateParameterNameNetwork(networkIndex, l, "W")])
			#print(Wbackup[generateParameterNameNetwork(networkIndex, l, "W")])
			#exit()

def relu(Z):
	
	A = tf.nn.relu(Z)
	
	return A

  
