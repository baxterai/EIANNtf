# -*- coding: utf-8 -*-
"""SANItf2_algorithmANN.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Usage:
see SANItf2.py

# Description

Define fully connected artificial neural network (ANN)

- Author: Richard Bruce Baxter - Copyright (c) 2020 Baxter AI (baxterai.com)

"""

import tensorflow as tf
import numpy as np
from SANItf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import SANItf2_globalDefs

W = {}
B = {}


#Network parameters
n_h = []
numberOfLayers = 0
numberOfNetworks = 0


def defineTrainingParametersANN(dataset, trainMultipleFiles):
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
		batchSize = 1000
	displayStep = 100
	
	return learningRate, trainingSteps, batchSize, displayStep
	

def defineNetworkParametersANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, numberOfNetworksSet):

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
		elif(dataset == "NewThyroid"):
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
		elif(dataset == "NewThyroid"):
			n_h_1 = 4
			n_h_2 = 4
		else:
			print("dataset unsupported")
			exit()
		n_h_3 = n_y
		n_h = [n_h_0, n_h_1, n_h_2, n_h_3]
		numberOfLayers = 3
		
	return numberOfLayers
	
	
def neuralNetworkPropagationANN(x, networkIndex=1):
			
	#print("numberOfLayers", numberOfLayers)
	
	AprevLayer = x
	for l in range(1, numberOfLayers+1):
		#print("l = " + str(l))
		Z = tf.add(tf.matmul(AprevLayer, W[generateParameterNameNetwork(networkIndex, l, "W")]), B[generateParameterNameNetwork(networkIndex, l, "B")])
		A = tf.nn.sigmoid(Z)
		AprevLayer = A
	
	return tf.nn.softmax(Z)

def defineNeuralNetworkParametersANN():

	print("numberOfNetworks", numberOfNetworks)
	
	randomNormal = tf.initializers.RandomNormal()
	
	for networkIndex in range(1, numberOfNetworks+1):
			
		for l in range(1, numberOfLayers+1):

			W[generateParameterNameNetwork(networkIndex, l, "W")] = tf.Variable(randomNormal([n_h[l-1], n_h[l]]))
			B[generateParameterNameNetwork(networkIndex, l, "B")] = tf.Variable(tf.zeros(n_h[l]))


