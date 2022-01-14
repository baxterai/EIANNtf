"""ANNtf2_algorithmANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2022 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNtf2.py

# Usage:
see ANNtf2.py

# Description:
ANNtf algorithm ANN - define artificial neural network

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs

supportSkipLayers = True

debugOnlyTrainFinalLayer = False
debugSingleLayerNetwork = False

debugFastTrain = False

W = {}
B = {}
if(supportSkipLayers):
	Ztrace = {}
	Atrace = {}

#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0

batchSize = 0

def defineTrainingParameters(dataset):
	global batchSize
	
	learningRate = 0.001
	batchSize = 100
	numEpochs = 10	#100 #10
	if(debugFastTrain):
		trainingSteps = batchSize
	else:
		trainingSteps = 10000	#1000

	displayStep = 100
			
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	


def defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	
	#numberOfLayers = defineNetworkParametersANNlegacy(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet)
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, numberOfNetworksSet, generateLargeNetwork=False)
	
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
	
	print("defineNetworkParametersANNlegacy, n_h = ", n_h)
	
	return numberOfLayers
	

def defineNeuralNetworkParameters():

	print("numberOfNetworks", numberOfNetworks)
	
	randomNormal = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):
			
		for l1 in range(1, numberOfLayers+1):
		
			if(supportSkipLayers):
				for l2 in range(0, l1):
					if(l2 < l1):
						Wlayer = randomNormal([n_h[l2], n_h[l1]]) 
						W[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "W")] = tf.Variable(Wlayer)
			else:	
				Wlayer = tf.Variable(randomNormal([n_h[l1-1], n_h[l1]]))
				W[generateParameterNameNetwork(networkIndex, l1, "W")] = Wlayer
			B[generateParameterNameNetwork(networkIndex, l1, "B")] = tf.Variable(tf.zeros(n_h[l1]))

			if(supportSkipLayers):
				Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))
				Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = tf.Variable(tf.zeros([batchSize, n_h[l1]], dtype=tf.dtypes.float32))

			#print("Wlayer = ", W[generateParameterNameNetwork(networkIndex, l1, "W")])

def neuralNetworkPropagation(x, networkIndex=1):
	return neuralNetworkPropagationANN(x, networkIndex)
	
def neuralNetworkPropagationANN(x, networkIndex=1):
			
	#print("numberOfLayers", numberOfLayers)
	
	AprevLayer = x
	if(supportSkipLayers):
		Atrace[generateParameterNameNetwork(networkIndex, 0, "Atrace")] = AprevLayer
	
	for l1 in range(1, numberOfLayers+1):
		if(supportSkipLayers):
			Z = tf.zeros(Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")].shape)
			for l2 in range(0, l1):
				Wlayer = W[generateParameterNameNetworkSkipLayers(networkIndex, l2, l1, "W")]
				at = Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")]
				Z = tf.add(Z, tf.add(tf.matmul(Atrace[generateParameterNameNetwork(networkIndex, l2, "Atrace")], Wlayer), B[generateParameterNameNetwork(networkIndex, l1, "B")]))		
		else:
			Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l1, "W")]), B[generateParameterNameNetwork(networkIndex, l1, "B")])
		
		A = activationFunction(Z)

		#print("l1 = " + str(l1))		
		#print("W = ", W[generateParameterNameNetwork(networkIndex, l1, "W")] )
		
		if(debugOnlyTrainFinalLayer):
			if(l1 < numberOfLayers):
				A = tf.stop_gradient(A)

		Ztrace[generateParameterNameNetwork(networkIndex, l1, "Ztrace")] = Z
		Atrace[generateParameterNameNetwork(networkIndex, l1, "Atrace")] = A
						
		AprevLayer = A
	
	return tf.nn.softmax(Z)


def activationFunction(Z):
	A = tf.nn.relu(Z)
	#A = tf.nn.sigmoid(Z)
	return A
	

