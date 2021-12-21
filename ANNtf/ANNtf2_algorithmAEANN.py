"""ANNtf2_algorithmAEANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2021 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNtf2.py

# Usage:
see ANNtf2.py

# Description:
ANNtf algorithm AEANN - define autoencoder artificial neural network

Greedy layer construction using autoencoder nonlinear dimensionality reduction

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs
import copy

debugFastTrain = False
debugSmallBatchSize = False	#small batch size for debugging matrix output

largeBatchSize = True	#train each layer using entire training set
generateLargeNetwork = False	#CHECKTHIS: network requires bottleneck 
generateNetworkStatic = False

	
#forward excitatory connections;
Wf = {}
Wb = {}
B = {}
	
#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0


#note high batchSize is required for learningAlgorithmStochastic algorithm objective functions (>= 100)
def defineTrainingParameters(dataset):
	global learningRate
	global weightDecayRate	
	
	learningRate = 0.001
	if(debugSmallBatchSize):
		batchSize = 10
	else:
		if(largeBatchSize):
			batchSize = 1000	#current implementation: batch size should contain all examples in training set
		else:
			batchSize = 100	#3	#100
	numEpochs = 10	#100 #10
	if(debugFastTrain):
		trainingSteps = batchSize
	else:
		trainingSteps = 10000	#1000
		
	displayStep = 100
			
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	


def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks

	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet, generateLargeNetwork=generateLargeNetwork, generateNetworkStatic=generateNetworkStatic)
			
	return numberOfLayers
	

def defineNeuralNetworkParameters():

	print("numberOfNetworks", numberOfNetworks)
	
	randomNormal = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):
		for l in range(1, numberOfLayers+1):
			#forward excitatory connections;
			WlayerF = randomNormal([n_h[l-1], n_h[l]]) 
			WlayerB = randomNormal([n_h[l], n_h[l-1]]) 
			Blayer = tf.zeros(n_h[l])
			Wf[generateParameterNameNetwork(networkIndex, l, "Wf")] = tf.Variable(WlayerF)
			Wb[generateParameterNameNetwork(networkIndex, l, "Wb")] = tf.Variable(WlayerB)
			B[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(Blayer)


def neuralNetworkPropagation(x, networkIndex=1):
	return neuralNetworkPropagationAEANNtest(x, networkIndex=1)

def neuralNetworkPropagationAEANNautoencoderLayer(x, layer, networkIndex=1):
	return neuralNetworkPropagationAEANN(x, trainAutoencoder=True, layer=layer, networkIndex=networkIndex)

def neuralNetworkPropagationAEANNfinalLayer(x, networkIndex=1):
	return neuralNetworkPropagationAEANN(x, trainAutoencoder=False, layer=numberOfLayers, networkIndex=networkIndex)
	
def neuralNetworkPropagationAEANNtest(x, networkIndex=1):
	return neuralNetworkPropagationAEANN(x, trainAutoencoder=False, layer=None, networkIndex=networkIndex)

def neuralNetworkPropagationAEANNtestLayer(x, l, networkIndex=1):
	return neuralNetworkPropagationAEANN(x, trainAutoencoder=False, layer=l, networkIndex=networkIndex)


def neuralNetworkPropagationAEANN(x, trainAutoencoder, layer, networkIndex=1):

	AprevLayer = x
	
	if(trainAutoencoder):
		maxLayer = layer
	else:
		if(layer is None):
			maxLayer = numberOfLayers
		else:
			maxLayer = layer
	
	output = x #in case layer=0
	
	for l in range(1, maxLayer+1):
	
		WlayerF = Wf[generateParameterNameNetwork(networkIndex, l, "Wf")]
		Z = tf.add(tf.matmul(AprevLayer, WlayerF), B[generateParameterNameNetwork(networkIndex, l, "B")])
		A = activationFunction(Z)

		if(trainAutoencoder):
			if(l == numberOfLayers):
				output = tf.nn.softmax(Z)
			else:
				if(l == layer):				
					#go backwards
					WlayerB = Wb[generateParameterNameNetwork(networkIndex, l, "Wb")]
					Z = tf.matmul(A, WlayerB)
					output = tf.nn.sigmoid(Z)	
		else:
			if(l == numberOfLayers):
				output = tf.nn.softmax(Z)
			else:
				output = A
				
		A = tf.stop_gradient(A)	#only train weights for layer l

		AprevLayer = A

	return output



def activationFunction(Z):
	A = tf.nn.relu(Z)
	return A
	
