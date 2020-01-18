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



W = {}
B = {}


#Network parameters
n_h = []
numberOfLayers = 0
numberOfSequentialInputs = 0


def defineNetworkParametersANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset):

	global n_h
	global numberOfLayers
	global numberOfSequentialInputs

	n_x = num_input_neurons #datasetNumFeatures
	n_y = num_output_neurons  #datasetNumClasses
	n_h_0 = n_x
	if(dataset == "POStagSequence"):
		n_h_1 = int(datasetNumFeatures*3) # 1st layer number of neurons.
		n_h_2 = int(datasetNumFeatures/2) # 2nd layer number of neurons.
	elif(dataset == "NewThyroid"):
		n_h_1 = 4
		n_h_2 = 4
	n_h_3 = n_y
	n_h = [n_h_0, n_h_1, n_h_2, n_h_3]
	numberOfLayers = 3
	numberOfSequentialInputs = 3
	
def neuralNetworkPropagationANN(x):

	global n_h
	global numberOfLayers
	global numberOfSequentialInputs
	
	AprevLayer = x
	for l in range(1, numberOfLayers+1):
		#print("l = " + str(l))
		Z = tf.add(tf.matmul(AprevLayer, W[generateParameterName(l, "W")]), B[generateParameterName(l, "B")])
		A = tf.nn.sigmoid(Z)
		AprevLayer = A
	
	return tf.nn.softmax(Z)

def defineNeuralNetworkParametersANN():

	randomNormal = tf.initializers.RandomNormal()

	global n_h
	global numberOfLayers
	global numberOfSequentialInputs
	
	for l in range(1, numberOfLayers+1):

		W[generateParameterName(l, "W")] = tf.Variable(randomNormal([n_h[l-1], n_h[l]]))
		B[generateParameterName(l, "B")] = tf.Variable(tf.zeros(n_h[l]))


