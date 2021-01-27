# -*- coding: utf-8 -*-
"""ANNtf2_algorithmANN.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Usage:
see ANNtf2.py

# Description

Define fully connected artificial neural network (ANN)

- Author: Richard Bruce Baxter - Copyright (c) 2020-2021 Baxter AI (baxterai.com)

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs

debugOnlyTrainFinalLayer = False
debugSingleLayerNetwork = False

debugFastTrain = False

W = {}
B = {}


#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0


def defineTrainingParametersANN(dataset, trainMultipleFiles):
	if(trainMultipleFiles):
		learningRate = 0.0001
		batchSize = 100
		numEpochs = 10
		if(dataset == "POStagSequence"):
			trainingSteps = 10000
		elif(dataset == "SmallDataset"):
			if(debugFastTrain):
				trainingSteps = batchSize
			else:
				trainingSteps = 10000	#1000

	else:
		learningRate = 0.001
		batchSize = 1
		numEpochs = 100 #10
		if(dataset == "POStagSequence"):
			trainingSteps = 10000
		elif(dataset == "SmallDataset"):
			if(debugFastTrain):
				trainingSteps = batchSize
			else:
				trainingSteps = 10000	#1000
		
	displayStep = 100
			
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	


def defineNetworkParametersANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet)
	#numberOfLayers = defineNetworkParametersANNlegacy(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet)
	
	return numberOfLayers
	
	
def defineNetworkParametersANNlegacy(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	numberOfNetworks = numberOfNetworksSet
	
	if(trainMultipleFiles):
		n_x = num_input_neurons #datasetNumFeatures
		n_y = num_output_neurons  #datasetNumClasses
		n_h_0 = n_x
		if(dataset == "POStagSequence"):
			n_h_1 = int(datasetNumFeatures*2)
			n_h_2 = int(datasetNumFeatures*1.5)
			n_h_3 = int(datasetNumFeatures)
			n_h_4 = int(datasetNumFeatures/2)
			n_h_5 = int(datasetNumFeatures/4)
		elif(dataset == "SmallDataset"):
			n_h_1 = 4
			n_h_2 = 4
		else:
			print("dataset unsupported")
			exit()
		n_h_6 = n_y
		n_h = [n_h_0, n_h_1, n_h_2, n_h_3, n_h_4, n_h_5, n_h_6]
		numberOfLayers = 6
	else:
		n_x = num_input_neurons #datasetNumFeatures
		n_y = num_output_neurons  #datasetNumClasses
		n_h_0 = n_x
		if(dataset == "POStagSequence"):
			n_h_1 = int(datasetNumFeatures*3) # 1st layer number of neurons.
			n_h_2 = int(datasetNumFeatures/2) # 2nd layer number of neurons.
		elif(dataset == "SmallDataset"):
			n_h_1 = 4
			n_h_2 = 4
		else:
			print("dataset unsupported")
			exit()
		n_h_3 = n_y
		if(debugSingleLayerNetwork):
			n_h = [n_h_0, n_h_3]	
		else:
			n_h = [n_h_0, n_h_1, n_h_2, n_h_3]
		numberOfLayers = len(n_h)-1
		
	return numberOfLayers
	

def defineNeuralNetworkParametersANN():

	print("numberOfNetworks", numberOfNetworks)
	
	randomNormal = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):
			
		for l in range(1, numberOfLayers+1):
			
			W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(randomNormal([n_h[l-1], n_h[l]]))
			B[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(tf.zeros(n_h[l]))

	
def neuralNetworkPropagationANN(x, networkIndex=1):
			
	#print("numberOfLayers", numberOfLayers)
	
	AprevLayer = x
	for l in range(1, numberOfLayers+1):
		Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
		A = activationFunction(Z)

		#print("l = " + str(l))		
		#print("W = ", W[generateParameterNameNetwork(networkIndex, l, "W")] )
		
		if(debugOnlyTrainFinalLayer):
			if(l < numberOfLayers):
				A = tf.stop_gradient(A)
				
		AprevLayer = A
	
	return tf.nn.softmax(Z)


def activationFunction(Z):
	A = tf.nn.relu(Z)
	#A = tf.nn.sigmoid(Z)
	return A
	

