"""ANNtf2_algorithmEIANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2021 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNtf2.py

# Usage:
see ANNtf2.py

# Description:
ANNtf algorithm EIANN - define excitatory/inhibitory artificial neural network

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs

learningAlgorithmFinalLayerBackpropHebbian = False	#only apply backprop to final layer (all intermediary layers using hebbian algorithm)


debugSingleLayerNetwork = False
debugFastTrain = False
debugGenerateLargeNetwork = True	#currently used to increase number of neurons per layer, as only 50% are excitatory 

W = {}
B = {}
neuronEI = {}	#tf.dtypes.bool


#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0
learningRate = 0.0

if(learningAlgorithmFinalLayerBackpropHebbian):
	useZAcoincidenceMatrix = True	#mandatory
	onlyUpdatePositiveWeights = True
	#flattenInhibitoryWeights #unimplemented - equivalent to a single inhibitory neuron per layer with equal weights/effect on each neuron

def defineTrainingParametersEIANN(dataset):
	global learningRate
	learningRate = 0.001
	batchSize = 100
	numEpochs = 10	#100 #10
	if(debugFastTrain):
		trainingSteps = batchSize
	else:
		trainingSteps = 10000	#1000
		
	displayStep = 100
			
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	


def defineNetworkParametersEIANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet, generateLargeNetwork=debugGenerateLargeNetwork)
	
	return numberOfLayers
	

def defineNeuralNetworkParametersEIANN():

	print("numberOfNetworks", numberOfNetworks)
	
	randomNormal = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):

		for l in range(0, numberOfLayers+1):	#last layer is ignored
			if(l == 0):	#not used
				neuronEIint = tf.ones([n_h[l]], dtype=tf.dtypes.int32)	#first layer is always excitatory
			else:	
				neuronEIint = tf.random.uniform([n_h[l]], minval=0, maxval=2, dtype=tf.dtypes.int32)
			neuronEI[generateParameterNameNetwork(networkIndex, l, "neuronEI")] = tf.Variable(tf.dtypes.cast(neuronEIint, dtype=tf.dtypes.bool))	#neuronEIint	#tf.dtypes.cast(neuronEIint, dtype=tf.dtypes.bool)	tf.dtypes.cast(neuronEIint, dtype=tf.dtypes.float32)

		#for l in range(1, numberOfLayers+1):	#first layer has no type enforcement
		#	neuronEIint = tf.random.uniform([n_h[l]], minval=0, maxval=2, dtype=tf.dtypes.int32)
		#	#print("neuronEIint.shape = ", neuronEIint.shape)
		#	neuronEI[generateParameterNameNetwork(networkIndex, l, "neuronEI")] = tf.Variable(tf.dtypes.cast(neuronEIint, dtype=tf.dtypes.bool))	#neuronEIint	#tf.dtypes.cast(neuronEIint, dtype=tf.dtypes.bool)	tf.dtypes.cast(neuronEIint, dtype=tf.dtypes.float32)

		for l in range(1, numberOfLayers+1):
		
			#set weights to positive/negative depending on the neuron type of the preceeding layer
				
			#if(l > 1):
			
			neuronEIprevious = neuronEI[generateParameterNameNetwork(networkIndex, l-1, "neuronEI")]

			Wlayer = randomNormal([n_h[l-1], n_h[l]])
			WlayerSign = tf.sign(Wlayer)
			WlayerSignBool = convertSignOutputToBool(WlayerSign)

			neuronEIpreviousTiled = tileDimension(neuronEIprevious, 1, n_h[l], True)
			WlayerSignCorrect = tf.equal(WlayerSignBool, neuronEIpreviousTiled)

			WlayerSignCorrect = tf.dtypes.cast(WlayerSignCorrect, dtype=tf.dtypes.float32)
			WlayerSignCorrectionFactor = tf.multiply(WlayerSignCorrect, 2.0)
			WlayerSignCorrectionFactor = tf.subtract(WlayerSignCorrectionFactor, 1.0)
			Wlayer = tf.multiply(Wlayer, WlayerSignCorrectionFactor)

			W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(Wlayer)
			B[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(tf.zeros(n_h[l]))

			#print("neuronEIprevious = ", neuronEIprevious)			
			#print("Wlayer = ", Wlayer)			
			
			#else:
			#	W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(randomNormal([n_h[l-1], n_h[l]]))
			#	B[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(tf.zeros(n_h[l]))	
				
			
#hebbian learning algorithm
def neuralNetworkPropagationEIANNtrain(x, networkIndex=1):
			
	#print("numberOfLayers", numberOfLayers)
	
	AprevLayer = x
	ZprevLayer = x
	for l in range(1, numberOfLayers+1):
	
		if(l < numberOfLayers):
				
			Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
			A = activationFunction(Z)
			
			AW = W[generateParameterNameNetwork(networkIndex, l, "W")]
			
			if(useZAcoincidenceMatrix):
				AWcontribution = tf.matmul(tf.transpose(ZprevLayer), A)	#increase excitatory weights that contributed to the output signal	#hebbian
			else:
				print("ANNtf2_algorithmEIANN error: useZAcoincidenceMatrix is currently required")
				AWcontribution = tf.matmul(tf.transpose(AprevLayer), A)	#increase excitatory weights that contributed to the output signal	#hebbian
				
			AWupdate = tf.multiply(AWcontribution, learningRate)

			if(onlyUpdatePositiveWeights):
				neuronEIprevious = neuronEI[generateParameterNameNetwork(networkIndex, l-1, "neuronEI")]
				neuronEIprevious = tf.dtypes.cast(neuronEIprevious, dtype=tf.dtypes.float32)
				neuronEIprevious = tf.expand_dims(neuronEIprevious, axis=1)	#for broadcasting
				AWupdate = tf.multiply(neuronEIprevious, AWupdate)	#broadcasting required	#zero all weight updates corresponding to inhibitory input

			AW = tf.add(AW, AWupdate)	#apply weight updates	

			W[generateParameterNameNetwork(networkIndex, l, "W")] = AW

			#print("l = " + str(l))		
			#print("W = ", W[generateParameterNameNetwork(networkIndex, l, "W")] )

			AprevLayer = A
			ZprevLayer = Z
					
	#return tf.nn.softmax(Z)
	
				
def neuralNetworkPropagationEIANN(x, networkIndex=1):
			
	#print("numberOfLayers", numberOfLayers)
	
	AprevLayer = x
	for l in range(1, numberOfLayers+1):
		Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
		A = activationFunction(Z)

		#print("l = " + str(l))		
		#print("W = ", W[generateParameterNameNetwork(networkIndex, l, "W")] )
		
		if(learningAlgorithmFinalLayerBackpropHebbian):
			if(l < numberOfLayers):
				A = tf.stop_gradient(A)
				
		AprevLayer = A
	
	return tf.nn.softmax(Z)


def activationFunction(Z):
	A = tf.nn.relu(Z)
	#A = tf.nn.sigmoid(Z)
	return A
	

