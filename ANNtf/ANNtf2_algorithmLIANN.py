"""ANNtf2_algorithmLIANN.py

# Author:
Richard Bruce Baxter - Copyright (c) 2020-2021 Baxter AI (baxterai.com)

# License:
MIT License

# Installation:
see ANNtf2.py

# Usage:
see ANNtf2.py

# Description:
ANNtf algorithm LIANN - define local inhibition artificial neural network (force neural independence)

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import ANNtf2_operations
import ANNtf2_globalDefs
import copy

onlyTrainFinalLayer = True	#only apply backprop (effective hebbian) learning at final layer
applyInhibitoryNetworkDuringTest = True
positiveExcitatoryWeights = True	#only allow positive excitatory neuron weights
#supportSkipLayers = True #fully connected skip layer network	#TODO: add support for skip layers	#see ANNtf2_algorithmFBANN for template

debugSingleLayerNetwork = False
debugFastTrain = False

#forward excitatory connections;
W = {}
B = {}
#lateral inhibitory connections (incoming/outgoing);
IWi = {}
IBi = {}
IWo = {}
IBiWeights = 0.3	#need at least 1/IBiWeights active neurons per layer for the inhibitory neuron to become activated
IBoWeights = -1.0	#inhibition contributes a significant (nullifying) effect on layer activation

#Network parameters
n_h = []
In_h = []
numberOfLayers = 0
numberOfNetworks = 0


def defineTrainingParametersLIANN(dataset, trainMultipleFiles):
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
		batchSize = 100
		numEpochs = 10	#100 #10
		if(dataset == "POStagSequence"):
			trainingSteps = 10000
		elif(dataset == "SmallDataset"):
			if(debugFastTrain):
				trainingSteps = batchSize
			else:
				trainingSteps = 10000	#1000
		
	displayStep = 100
			
	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	


def defineNetworkParametersLIANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet):

	global n_h
	global numberOfLayers
	global numberOfNetworks
	global In_h
	
	n_h, numberOfLayers, numberOfNetworks, datasetNumClasses = ANNtf2_operations.defineNetworkParameters(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet)
	
	In_h = copy.copy(n_h)	#create one inhibitory neuron for every excitatory neuron
	
	return numberOfLayers
	

def defineNeuralNetworkParametersLIANN():

	print("numberOfNetworks", numberOfNetworks)
	
	randomNormal = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):
		for l in range(1, numberOfLayers+1):
			#forward excitatory connections;
			EWlayer = randomNormal([n_h[l-1], n_h[l]])
			if(positiveExcitatoryWeights):
				EWlayer = tf.abs(EWlayer)
			W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(EWlayer)
			B[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(tf.zeros(n_h[l]))

			#lateral inhibitory connections (incoming/outgoing);
			#do not currently train inhibitory weights;
			IWilayer = tf.multiply(tf.ones([n_h[l], In_h[l]]), IBiWeights)		#CHECKTHIS: inhibitory neuron firing is a function of current (lateral) layer (not previous layer)
			IWolayer = tf.multiply(tf.ones([In_h[l], n_h[l]]), IBoWeights)
			IWi[generateParameterNameNetwork(networkIndex, l, "IWi")] = tf.Variable(IWilayer)
			IBi[generateParameterNameNetwork(networkIndex, l, "IBi")] = tf.Variable(tf.zeros(In_h[l]))
			IWo[generateParameterNameNetwork(networkIndex, l, "IWo")] = tf.Variable(IWolayer)
			

def neuralNetworkPropagationLIANNtrain(x, y, networkIndex=1):
	AprevLayer = x
	for l in range(1, numberOfLayers+1):
		if(l < numberOfLayers):
			#TODO: determine learning algorithm (how to modify weights to maximise independence between neurons on each layer)
			
			#forward excitatory connections;
			Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
			A = activationFunction(Z)
			
			#lateral inhibitory connections (incoming/outgoing);
			IZi = tf.matmul(A, IWi[generateParameterNameNetwork(networkIndex, l, "IWi")])	#CHECKTHIS: inhibitory neuron firing is a function of current (lateral) layer (not previous layer)
			IAi = activationFunction(IZi)
			IZo = tf.matmul(IAi, IWo[generateParameterNameNetwork(networkIndex, l, "IWo")])

			Zmod = tf.add(Z, IZo)
			Amod = activationFunction(Zmod)
			
			AprevLayer = Amod

	
def neuralNetworkPropagationLIANN(x, networkIndex=1):
			
	#print("numberOfLayers", numberOfLayers)
	
	AprevLayer = x
	for l in range(1, numberOfLayers+1):
		if(l < numberOfLayers):
			#forward excitatory connections;
			Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
			A = activationFunction(Z)
			
			if(applyInhibitoryNetworkDuringTest):
				#lateral inhibitory connections (incoming/outgoing);
				IZi = tf.matmul(A, IWi[generateParameterNameNetwork(networkIndex, l, "IWi")])	#CHECKTHIS: inhibitory neuron firing is a function of current (lateral) layer (not previous layer)
				IAi = activationFunction(IZi)
				IZo = tf.matmul(IAi, IWo[generateParameterNameNetwork(networkIndex, l, "IWo")])

				Zmod = tf.add(Z, IZo)
				Amod = activationFunction(Zmod)
			else:
				Amod = A
				
			if(onlyTrainFinalLayer):
				Amod = tf.stop_gradient(Amod)
			AprevLayer = Amod
			
		else:
			#final layer (do not apply inhibition)
			Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
			A = activationFunction(Z)	
					
	
	return tf.nn.softmax(Z)


def activationFunction(Z):
	A = tf.nn.relu(Z)
	#A = tf.nn.sigmoid(Z)
	return A
	

