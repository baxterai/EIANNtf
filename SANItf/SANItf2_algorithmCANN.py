# -*- coding: utf-8 -*-
"""SANItf2_algorithmCANN.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Usage:
see SANItf2.py

# Description

Define fully connected classification artificial neural network (CANN) - unsupervised hebbian learning

- Author: Richard Bruce Baxter - Copyright (c) 2020 Baxter AI (baxterai.com)

"""

import tensorflow as tf
import numpy as np
from SANItf2_operations import *	#generateParameterNameSeq, generateParameterName, defineNetworkParameters
import SANItf2_globalDefs


enableForgetting = False

W = {}
B = {}


#Network parameters
n_h = []
numberOfLayers = 0

learningRateLocal = 0.0
forgetRateLocal = 0.0

def defineTrainingParametersCANN(dataset, trainMultipleFiles):
	if(trainMultipleFiles):
		learningRate = 0.0001
		forgetRate = 0.00001
		if(dataset == "POStagSequence"):
			trainingSteps = 10000
		elif(dataset == "NewThyroid"):
			trainingSteps = 1000
		batchSize = 100
	else:
		learningRate = 0.001
		forgetRate = 0.0001
		if(dataset == "POStagSequence"):
			trainingSteps = 10000
		elif(dataset == "NewThyroid"):
			trainingSteps = 1000
		batchSize = 1000
	displayStep = 100
	
	learningRateLocal = learningRate
	forgetRateLocal = forgetRate
	
	return learningRate, trainingSteps, batchSize, displayStep
	

def defineNetworkParametersCANN(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles):

	global n_h
	global numberOfLayers
	
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
			n_h_1 = 10
			n_h_2 = 9
			n_h_3 = 8
			n_h_4 = 7
			n_h_5 = 6
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
	
	
def neuralNetworkPropagationTrainCANN(x, train=False):
		
	AprevLayer = x
	 
	for l in range(1, numberOfLayers+1):
	
		#print("l = " + str(l))

		Z = tf.add(tf.matmul(AprevLayer, W[generateParameterName(l, "W")]), B[generateParameterName(l, "B")])
		A = tf.nn.relu(Z)
		
		if(train):
			#update weights based on hebbian learning rule
			#strengthen those connections that caused the current layer neuron to fire (and weaken those that did not)

			AcorrelationMatrix = tf.matmul(tf.transpose(AprevLayer), A)
			#Bmod = 0*learningRate	#biases are not currently used
			Wmod = AcorrelationMatrix*learningRateLocal
			#B[generateParameterName(l, "B")] = B[generateParameterName(l, "B")] + Bmod
			W[generateParameterName(l, "W")] = W[generateParameterName(l, "W")] + Wmod
				
			if(enableForgetting):
				AcorrelationMatrixIsZero = tf.math.equal(AcorrelationMatrix, 0)
				AcorrelationMatrixIsZeroInt = tf.dtypes.cast(AcorrelationMatrixIsZero, tf.int32)
				AcorrelationMatrixIsZeroFloat = tf.dtypes.cast(AcorrelationMatrixIsZeroInt, dtype=tf.float32)
				Wmod2 = AcorrelationMatrixIsZeroFloat*-forgetRateLocal
				W[generateParameterName(l, "W")] = W[generateParameterName(l, "W")] + Wmod2
						
		AprevLayer = A
			
	return tf.nn.softmax(Z)	#not used


def defineNeuralNetworkParametersCANN():

	randomNormal = tf.initializers.RandomNormal()
	
	for l in range(1, numberOfLayers+1):

		W[generateParameterName(l, "W")] = tf.Variable(randomNormal([n_h[l-1], n_h[l]]))
		B[generateParameterName(l, "B")] = tf.Variable(tf.zeros(n_h[l]))


