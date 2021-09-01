# -*- coding: utf-8 -*-
"""ANNtf2_algorithmSANI.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Usage:
see ANNtf2.py

# Description:

Define Sequentially Activated Neuronal Input (SANI) neural net - common code for reference

- Author: Richard Bruce Baxter - Copyright (c) 2020-2021 Baxter AI (baxterai.com)

"""

#start common ANNtf2_algorithmSANI.py code:

import tensorflow as tf
import numpy as np
from ANNtf2_operations import * #generateParameterNameSeq, generateParameterName
import ANNtf2_algorithmSANIoperations
from ANNtf2_algorithmSANIglobalDefs import *
import ANNtf2_globalDefs


#parameters
#static parameters (convert from tf.variable to tf.constant?):
Cseq = {}
CseqLayer = {}	
n_h_cumulative = {}
#variable parameters:
WRseq = {}	#weights matrix
WR = {}	#weights matrix
BR = {}	#biases vector
Wseq = {}	#weights matrix
Bseq = {}	#biases vector
W = {}	#weights matrix
B = {}	#biases vector

#parameters
#static parameters (convert from tf.variable to tf.constant?):
#if(not supportFullConnectivity):
#	if(useSparseTensors):
#		Cseq = {}	#connectivity vector
#		if(supportSkipLayers):	
#			CseqLayer = {}	
#			n_h_cumulative = {}
##variable parameters:	
#if((algorithmSANI == "sharedModulesHebbian") or (algorithmSANI == "sharedModulesBinary") or (algorithmSANI == "sharedModules")):
#	if(recordNetworkWeights):
#		if(recordSubInputsWeighted):
#			AseqInputVerified = {}
#			WRseq = {}	#weights matrix
#		if(recordSequentialInputsWeighted):
#			WR = {}	#weights matrix
#		if(recordNeuronsWeighted):
#			BR = {}	#biases vector
#if((algorithmSANI == "sharedModulesHebbian") or (algorithmSANI == "sharedModules") or (algorithmSANI == "repeatedModules")):
#	#variable parameters (tf.variable): 
#	if(allowMultipleSubinputsPerSequentialInput):
#		if(performSummationOfSubInputsWeighted):
#			Wseq = {}	#weights matrix
#			Bseq = {}	#biases vector
#	if(performSummationOfSequentialInputsWeighted):
#		W = {}	#weights matrix
#		B = {}	#biases vector
	
			
#Network parameters
n_h = []
numberOfLayers = 0

#if((algorithmSANI == "sharedModulesHebbian") or (algorithmSANI == "sharedModulesBinary") or (algorithmSANI == "sharedModules")):	#only code to currently use these variables
numberOfFeaturesPerWord = -1
paddingTagIndex = -1
def defineTrainingParametersSANIsharedModules(numberOfFeaturesPerWordNew, paddingTagIndexNew):
	#if((algorithmSANI == "sharedModulesHebbian") or (algorithmSANI == "sharedModulesBinary") or (algorithmSANI == "sharedModules")):	#only code to currently use these variables
	global numberOfFeaturesPerWord
	global paddingTagIndex
	numberOfFeaturesPerWord = numberOfFeaturesPerWordNew
	paddingTagIndex = paddingTagIndexNew

def defineNetworkParametersSANIwrapper(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, useSmallSentenceLengths, numberOfFeaturesPerWord):
	global n_h
	global numberOfLayers
	n_h, numberOfLayers = ANNtf2_algorithmSANIoperations.defineNetworkParametersSANI(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, useSmallSentenceLengths, numberOfFeaturesPerWord)
	
def defineTrainingParametersSANIwrapper(dataset, trainMultipleFiles):
	return ANNtf2_algorithmSANIoperations.defineTrainingParametersSANI(dataset, trainMultipleFiles)
	

def defineNeuralNetworkParametersSANI():
	global n_h_cumulative
	ANNtf2_algorithmSANIoperations.defineNeuralNetworkParametersSANI(n_h, numberOfLayers, Cseq, CseqLayer, n_h_cumulative, WRseq, WR, BR, Wseq, Bseq, W, B)
			

#temporary variables for neuralNetworkPropagationSANI:
if(algorithmSANI == "sharedModulesHebbian"):
	Vseq = {}
	Zseq = {}
	Aseq = {}
	Z = {}
	A = {}
	sequentialActivationFound = {}	#CHECKTHIS: is this required?
	if(algorithmSANI == "sharedModulesHebbian"):
		if(useHebbianLearningRuleApply):
			WseqDelta = {}	#prospective weights update
elif(algorithmSANI == "sharedModulesBinary"):
	Vseq = {}
	Zseq = {}
	Aseq = {}
	TMaxSeq = {}
	TMinSeq = {}
	ZseqTadjusted = {}
	Z = {}
	A = {}
	T = {}
	TMax = {}
	TMin = {}
	sequentialActivationFound = {}
	AseqInputVerified = {}
elif(algorithmSANI == "sharedModules"):
	Vseq = {}
	Zseq = {}
	Aseq = {}
	TMaxSeq = {}
	TMinSeq = {}
	ZseqTadjusted = {}
	Z = {}
	A = {}
	TMax = {}
	TMin = {}
	sequentialActivationFound = {}
	AseqInputVerified = {}
elif(algorithmSANI == "repeatedModules"):
	pass

#end common ANNtf2_algorithmSANI.py code


def neuralNetworkPropagationSANI(x):
	if(algorithmSANI == "sharedModulesHebbian"):
		return neuralNetworkPropagationSANIsharedModulesHebbian(x)
	elif(algorithmSANI == "sharedModulesBinary"):
		return neuralNetworkPropagationSANIsharedModulesBinary(x)
	elif(algorithmSANI == "sharedModules"):
		return neuralNetworkPropagationSANIsharedModules(x)
	elif(algorithmSANI == "repeatedModules"):
		return neuralNetworkPropagationSANIrepeatedModules(x)

	
