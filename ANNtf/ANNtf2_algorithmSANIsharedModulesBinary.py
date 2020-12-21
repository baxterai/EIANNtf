# -*- coding: utf-8 -*-
"""ANNtf2_algorithmSANIsharedModulesBinary.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Usage:
see ANNtf2.py

# Description:

Define Sequentially Activated Neuronal Input (SANI) neural net - shared modules 

Neural modules can be shared between different areas of input sequence, eg sentence (cf RNN).

This code mirrors that of GIA Sequence Grammar ANN.
  
Can parse (by default expects to parse) full sentences; ie features for each word in sentence.

- Author: Richard Bruce Baxter - Copyright (c) 2020 Baxter AI (baxterai.com)

"""

import tensorflow as tf
import numpy as np
from ANNtf2_operations import * #generateParameterNameSeq, generateParameterName
import ANNtf2_globalDefs


numberOfFeaturesPerWord = -1
paddingTagIndex = -1




allowMultipleSubinputsPerSequentialInput = False

numberOfWordsInConvolutionalWindowSeen = 1	#always 1
	
resetSequentialInputsIfOnlyFirstInputValid = True	#see GIA_TXT_REL_TRANSLATOR_NEURAL_NETWORK_SEQUENCE_GRAMMAR development history for meaning and algorithmic implications of this feature
if(resetSequentialInputsIfOnlyFirstInputValid):
	doNotResetNeuronOutputUntilAllSequentialInputsActivated = True

useSparseTensors = True	#mandatory

enforceTcontiguityConstraints = True
if(enforceTcontiguityConstraints):
	enforceTcontiguityBetweenSequentialInputs = True
	enforceTcontiguityTakeEncapsulatedSubinputWithMinTvalue = True	#method to decide between subinput selection/parse tree generation
	enforceTcontiguityTakeEncapsulatedSubinputWithMaxTvalueEqualsW = True
	enforceTcontiguityStartAndEndOfSequence = True
	
expectNetworkConvergence = False
if(expectNetworkConvergence):
	#if(numberOfSequentialInputs == 2):
	oneSequentialInputHasOnlyOneSubinput = True	#conditional probability determination of events
	if(oneSequentialInputHasOnlyOneSubinput):
		firstSequentialInputHasOnlyOneSubinput = True #use combination of allowMultipleSubinputsPerSequentialInput for different sequential inputs;  #1[#2] sequential input should allow multiple subinputs, #2[#1] sequential input should allow single subinput
		if(firstSequentialInputHasOnlyOneSubinput):
			lastSequentialInputHasOnlyOneSubinput = False
		else:
			lastSequentialInputHasOnlyOneSubinput = True
else:
	oneSequentialInputHasOnlyOneSubinput = False

if(not ANNtf2_globalDefs.testHarness):	
	recordNetworkWeights = True
	if(recordNetworkWeights):
		recordSubInputsWeighted = True
		recordSequentialInputsWeighted = False	#may not be necessary (only used if can split neuron sequential inputs)
		recordNeuronsWeighted = True
		#FUTURE: prune network neurons/connections based on the relative strength of these weights
else:
	recordNetworkWeights = False
	recordSubInputsWeighted = False
	recordSequentialInputsWeighted = False
	recordNeuronsWeighted = False

if(ANNtf2_globalDefs.testHarness):
	supportSkipLayers = True
else:
	supportSkipLayers = True
if(expectNetworkConvergence):
	maxNumberSubinputsPerSequentialInput = 50	#~approx equal number of prev layer neurons/2 (FUTURE: make dynamic based on layer index)	#number of prior/future events in which to calculate a conditional probability
else:
	maxNumberSubinputsPerSequentialInput = 1	#sparsity
	
useLastSequentialInputOnly = True	#implied variable (not used)




#variable parameters (tf.variable):
if(recordNetworkWeights):
	if(recordSubInputsWeighted):
		AseqInputVerified = {}
		WRseq = {}	#weights matrix
	if(recordSequentialInputsWeighted):	
		WR = {}	#weights matrix
	if(recordNeuronsWeighted):
		BR = {}	#biases vector

	
		
#static parameters (convert from tf.variable to tf.constant?):
Cseq = {}	#connectivity vector
if(supportSkipLayers):	
	CseqLayer = {}
if(supportSkipLayers):
	n_h_cumulative = {}
		
		
#Network parameters
n_h = []
numberOfLayers = 0
numberOfSequentialInputs = 0

#temporary variables:
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

def defineTrainingParametersSANIsharedModules(numberOfFeaturesPerWordNew, paddingTagIndexNew):
	global numberOfFeaturesPerWord
	global paddingTagIndex
	numberOfFeaturesPerWord = numberOfFeaturesPerWordNew
	paddingTagIndex = paddingTagIndexNew

def defineTrainingParametersSANI(dataset, trainMultipleFiles):
	#Training parameters
	if(ANNtf2_globalDefs.testHarness):	
		learningRate = 0.001
		trainingSteps = 1
		numEpochs = 1
	else:
		if(trainMultipleFiles):
			learningRate = 0.001
			if(dataset == "POStagSentence"):
				trainingSteps = 10000
			numEpochs = 10
		else:
			learningRate = 0.001
			if(dataset == "POStagSentence"):
				trainingSteps = 10000
			numEpochs = 1

	batchSize = 1	#4	#32	#128	#256
	displayStep = 1

	return learningRate, trainingSteps, batchSize, displayStep, numEpochs
	
def defineNetworkParametersSANI(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, useSmallSentenceLengths):
	
	global n_h
	global numberOfLayers
	global numberOfSequentialInputs
	
	inputLength = numberOfFeaturesPerWord*numberOfWordsInConvolutionalWindowSeen

	n_x = inputLength #datasetNumFeatures
	n_y = 1  #SANIshared uses a single output neuron (either 1 or 0)	#if multiple output classes: n_y = num_output_neurons-1 or datasetNumClasses-1	
	n_h_0 = n_x
	if(dataset == "POStagSentence"):
		
		#FUTURE: the number of neurons/connections should be greatly increased, then pruned
		#FUTURE: upgrade to support multiple permutations

		#number of layers should approx = 2^maxNumWordsInSentence
		
		if(ANNtf2_globalDefs.testHarness):
			n_h_1 = int(1)
			n_h_2 = int(1)
			n_h = [n_h_0, n_h_1, n_h_2]
		else:
			if(useSmallSentenceLengths):
				#maximumSentenceLength = ~20 words
				if(expectNetworkConvergence):
					n_h_1 = int(inputLength*inputLength)	#*x for skip layers	
					n_h_2 = int(inputLength*inputLength)
					n_h_3 = int(inputLength*inputLength)
					n_h_4 = int(inputLength*inputLength)
					n_h_5 = n_y
				else:
					n_h_1 = int(inputLength*inputLength)
					n_h_2 = int(inputLength*inputLength*inputLength)
					n_h_3 = int(inputLength*inputLength*inputLength*inputLength)
					n_h_4 = int(inputLength*inputLength*inputLength*inputLength)
					n_h_5 = int(inputLength*inputLength*inputLength*inputLength)
					#n_h_1 = int(inputLength*inputLength)	#*x for skip layers	
					#n_h_2 = int(inputLength*inputLength)
					#n_h_3 = int(inputLength*inputLength*inputLength)
					#n_h_4 = int(inputLength*inputLength*inputLength)
					#n_h_5 = int(inputLength*inputLength*inputLength)				
				n_h = [n_h_0, n_h_1, n_h_2, n_h_3, n_h_4, n_h_5]

			else:
				#maximumSentenceLength = ~50 words
				if(expectNetworkConvergence):
					n_h_1 = int(inputLength*inputLength)	#*x for skip layers	
					n_h_2 = int(inputLength*inputLength)
					n_h_3 = int(inputLength*inputLength)
					n_h_4 = int(inputLength*inputLength)
					n_h_5 = int(inputLength*inputLength)
					n_h_6 = int(inputLength*inputLength)
					n_h_7 = n_y
				else:
					n_h_1 = int(inputLength*inputLength)	#*x for skip layers	
					n_h_2 = int(inputLength*inputLength)
					n_h_3 = int(inputLength*inputLength*inputLength)
					n_h_4 = int(inputLength*inputLength*inputLength)
					n_h_5 = int(inputLength*inputLength*inputLength*inputLength)
					n_h_6 = int(inputLength*inputLength*inputLength*inputLength)
					n_h_7 = int(inputLength*inputLength*inputLength*inputLength)
				n_h = [n_h_0, n_h_1, n_h_2, n_h_3, n_h_4, n_h_5, n_h_6, n_h_7]

			#n_h_1 = int(datasetNumFeatures*1)	#*x for skip layers	
			#n_h_2 = int(datasetNumFeatures*2)
			#n_h_3 = int(datasetNumFeatures*3)
			#n_h_4 = int(datasetNumFeatures*4)
			#n_h_5 = n_y
			#n_h = [n_h_0, n_h_1, n_h_2, n_h_3, n_h_4, n_h_5]

			#n_h_1 = int(datasetNumFeatures*3)
			#n_h_2 = int(datasetNumFeatures/2)
			#n_h_3 = n_y
			#n_h = [n_h_0, n_h_1, n_h_2, n_h_3]

			#n_h_1 = int(inputLength)
			#n_h_2 = int(inputLength/2)
			#n_h_3 = int(inputLength/4)
			#n_h_4 = int(inputLength/8)
			#n_h_5 = n_y
			#n_h = [n_h_0, n_h_1, n_h_2, n_h_3, n_h_4, n_h_5]

	else:
		print("sequential input data is required")
		exit()
		
	numberOfLayers = len(n_h)-1
	numberOfSequentialInputs = 2	#3
	

def defineNeuralNetworkParametersSANI():

	randomNormal = tf.initializers.RandomNormal()

	global n_h_cumulative
	
	if(supportSkipLayers):
		n_h_cumulativeNP = np.zeros((numberOfLayers+2), dtype=int)
		n_h_cumulativeNP[0] = 0	#first row always set to 0 for indexing purposes
		#print("n_h_cumulativeNP[0] = ", n_h_cumulativeNP[0])
		#print("\tn_h[0] = " + str(n_h[0]))

	if(ANNtf2_globalDefs.testHarness):
		if(supportSkipLayers):
			
			# simple net:
			#
			# L0: w0:1 0 [w1:1] 0 [w2:1] 0 0 0 0 0 . . . 0 0	features [53]
			#     |   /         /
			# L1: x  x      /
			#     |    /
			# L2: x  x
			
			numberSubinputsPerSequentialInput = calculateNumberSubinputsPerSequentialInput(0)
			
			CseqNPl1c0 = np.zeros((numberSubinputsPerSequentialInput, n_h[1]))
			CseqNPl1c1 = np.zeros((numberSubinputsPerSequentialInput, n_h[1]))
			CseqNPl2c0 = np.zeros((numberSubinputsPerSequentialInput, n_h[2]))
			CseqNPl2c1 = np.zeros((numberSubinputsPerSequentialInput, n_h[2]))
			CseqLayerNPl1c0 = np.zeros((numberSubinputsPerSequentialInput, n_h[1]))
			CseqLayerNPl1c1 = np.zeros((numberSubinputsPerSequentialInput, n_h[1]))
			CseqLayerNPl2c0 = np.zeros((numberSubinputsPerSequentialInput, n_h[1]))
			CseqLayerNPl2c1 = np.zeros((numberSubinputsPerSequentialInput, n_h[1]))
			
			CseqNPl1c0[0, 0] = 0
			CseqNPl1c1[0, 0] = 1
			CseqLayerNPl1c0[0, 0] = 0
			CseqLayerNPl1c1[0, 0] = 0
			n_h_cumulativeNP[1] = n_h_cumulativeNP[0] + n_h[0]
			
			CseqNPl2c0[0, 0] = 0
			CseqNPl2c1[0, 0] = 2
			CseqLayerNPl2c0[0, 0] = 1
			CseqLayerNPl2c1[0, 0] = 0
			n_h_cumulativeNP[2] = n_h_cumulativeNP[1] + n_h[1]
			
			Cseq[generateParameterNameSeq(1, 0, "Cseq")] = tf.Variable(CseqNPl1c0, dtype=tf.int32)
			Cseq[generateParameterNameSeq(1, 1, "Cseq")] = tf.Variable(CseqNPl1c1, dtype=tf.int32)
			Cseq[generateParameterNameSeq(2, 0, "Cseq")] = tf.Variable(CseqNPl2c0, dtype=tf.int32)
			Cseq[generateParameterNameSeq(2, 1, "Cseq")] = tf.Variable(CseqNPl2c1, dtype=tf.int32)
			CseqLayer[generateParameterNameSeq(1, 0, "CseqLayer")] = tf.Variable(CseqLayerNPl1c0, dtype=tf.int32)
			CseqLayer[generateParameterNameSeq(1, 1, "CseqLayer")] = tf.Variable(CseqLayerNPl1c1, dtype=tf.int32)
			CseqLayer[generateParameterNameSeq(2, 0, "CseqLayer")] = tf.Variable(CseqLayerNPl2c0, dtype=tf.int32)
			CseqLayer[generateParameterNameSeq(2, 1, "CseqLayer")] = tf.Variable(CseqLayerNPl2c1, dtype=tf.int32)
	else:
		for l in range(1, numberOfLayers+1):
			#print("\tl = " + str(l))
			#print("\tn_h[l] = " + str(n_h[l]))
			for s in range(numberOfSequentialInputs):
				#print("\t\ts = " + str(s))

				numberSubinputsPerSequentialInput = calculateNumberSubinputsPerSequentialInput(s)

				if(supportSkipLayers):
					#neuronIndex = np.random.randint(0, n_h_cumulativeNP[l]+1, n_h[l])
					CseqNP = np.zeros((numberSubinputsPerSequentialInput, n_h[l]))
					CseqLayerNP = np.random.randint(0, l, (numberSubinputsPerSequentialInput, n_h[l]))	#this can be modified to make local/distant connections more probable
					for i in range(numberSubinputsPerSequentialInput):
						for j in range(n_h[l]):
							l2 = CseqLayerNP[i, j]
							CseqNP[i,j] = np.random.randint(0, n_h[l2], 1)
					Cseq[generateParameterNameSeq(l, s, "Cseq")] = tf.Variable(CseqNP, dtype=tf.int32)
					CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")] = tf.Variable(CseqLayerNP, dtype=tf.int32)

					#printAverage(Cseq[generateParameterNameSeq(l, s, "Cseq")], "Cseq", 3)
					#printAverage(CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")], "CseqLayer", 3)

				else:
					CseqNP = np.random.randint(0, n_h[l-1]+1, (numberSubinputsPerSequentialInput, n_h[l]))	#note +1 is required because np.random.randint generates int between min and max-1
					Cseq[generateParameterNameSeq(l, s, "Cseq")] = tf.Variable(CseqNP, dtype=tf.int32)

				if(recordSubInputsWeighted):
					WRseqNP = np.random.rand(numberSubinputsPerSequentialInput, n_h[l])
					WRseq[generateParameterNameSeq(l, s, "WRseq")] = tf.Variable(WRseqNP, dtype=tf.float32)

			if(recordSequentialInputsWeighted):	
				WR[generateParameterName(l, "WR")] = tf.Variable(randomNormal([numberOfSequentialInputs, n_h[l]], dtype=tf.float32))	#randomNormal	#note the architecture of this weight matrix is different than a normal weight matrix. Every set of numberOfSequentialInputs (e.g. 3) represents a unique set of sequential neuron inputs (artificial neurons), and are mapped to an independent neuron (real neuron)
			if(recordNeuronsWeighted):		
				BR[generateParameterName(l, "BR")] = tf.Variable(tf.zeros(n_h[l]), tf.float32)

			if(supportSkipLayers):
				n_h_cumulativeNP[l] = n_h_cumulativeNP[l-1] + n_h[l-1]

	if(supportSkipLayers):
		n_h_cumulativeNP[numberOfLayers+1] = n_h_cumulativeNP[numberOfLayers] + n_h[numberOfLayers]	#not used
		
		n_h_cumulative['n_h_cumulative'] = tf.Variable(n_h_cumulativeNP, dtype=tf.int32)



def calculateNumberSubinputsPerSequentialInput(s):

	if(oneSequentialInputHasOnlyOneSubinput):
		if(firstSequentialInputHasOnlyOneSubinput and s==0):
			numberSubinputsPerSequentialInput = 1
		elif(lastSequentialInputHasOnlyOneSubinput and s==numberOfSequentialInputs-1):
			numberSubinputsPerSequentialInput = 1
		else:
			numberSubinputsPerSequentialInput = maxNumberSubinputsPerSequentialInput
	else:
		numberSubinputsPerSequentialInput = maxNumberSubinputsPerSequentialInput
	
	return numberSubinputsPerSequentialInput
					
					
						
def neuralNetworkPropagationSANI(x):
		
	#note connectivity indexes are used rather than sparse weight matrices due to limitations in current tf2 sparse tensor implementation
	
	#definitions for reference:
	
	#neuron sequential input vars;
	#x/AprevLayer	#output vector (dim: batchSize*n_h[l])

	#Cseq	#static connectivity matrix (int) - indexes of neurons on prior layer stored; mapped to W  (dim: maxNumberSubinputsPerSequentialInput*n_h[l])
	#if(supportSkipLayers):
		#CseqLayer	#static connectivity matrix (int) - indexes of pior layer stored; mapped to W (dim: maxNumberSubinputsPerSequentialInput*n_h[l])
	#AseqSum	#combination variable

	#Vseq	#mutable verification vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)			#records whether particular neuron sequential inputs are currently active
	#TMaxSeq	#mutable time vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)				#records the time at which a particular sequential input last fired
	#TMinSeq	#mutable time vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)				#records the time at which a first encapsulated subinput fired
	#Zseq	#neuron activation function output vector (dim: batchSize*n_h[l]  - regenerated for each sequential input index)	#records the current output value of sequential inputs
	#Aseq	#neuron activation function output vector (dim: batchSize*n_h[l]  - regenerated for each sequential input index)	#records the current output value of sequential inputs
	#if(recordSubInputsWeighted):
		#AseqInputVerified	#neuron input (sequentially verified) (dim: batchSize*numberSubinputsPerSequentialInput*n_h[l]  - records the subinputs that were used to activate the sequential input)
		#WRseq #weights of connections; see Cseq (dim: maxNumberSubinputsPerSequentialInput*n_h[l])
		
	#Q  
	#Z	#neuron activation function input (dim: batchSize*n_h[l])
	#A	#neuron activation function output (dim: batchSize*n_h[l])
	#TMax	#mutable time vector (dim: batchSize*n_h[l]) - same as TMaxSeq[numberOfSequentialInputs-1]
	#TMin	#mutable time vector (dim: batchSize*n_h[l]) - same as TMinSeq[numberOfSequentialInputs-1]
	#if(recordSequentialInputsWeighted):	
		#W	(dim: numberOfSequentialInputs*n_h[l])	
	#if(recordNeuronsWeighted):	
		#B	(dim: n_h[l])	
				
	batchSize = x.shape[0]
	
	#optimise feed length based on max sentence length in batch:
	#unoptimised: numberOfFeatures = x.shape[1]
	
	xIsNotPadding = tf.math.less(x, paddingTagIndex)
	coordinatesOfNotPadding = tf.where(xIsNotPadding)
	numberOfFeaturesCropped = tf.reduce_max(coordinatesOfNotPadding[:, 1]).numpy()+1

	#method 2 trial;
	#match_indices2 = tf.where(tf.equal(paddingTagIndex, x), x=tf.range(tf.shape(x)[1])*tf.ones_like(x), y=0*tf.ones_like(x))
	#print(match_indices2)
	#numberOfFeaturesCropped = tf.reduce_max(match_indices2)
	#print("numberOfFeaturesCropped = ", numberOfFeaturesCropped)

	inputLength = numberOfFeaturesPerWord*numberOfWordsInConvolutionalWindowSeen
	
	maxNumberOfWordsInSentenceBatch = int(numberOfFeaturesCropped/numberOfFeaturesPerWord)
	
	match_indices = tf.where(tf.equal(paddingTagIndex, x), x=tf.range(tf.shape(x)[1])*tf.ones_like(x), y=(tf.shape(x)[1])*tf.ones_like(x))
	numberOfFeaturesActiveBatch = tf.math.argmin(match_indices, axis=1).numpy()
	numberOfWordsInSentenceBatch = (numberOfFeaturesActiveBatch/numberOfFeaturesPerWord).astype(np.int32) 
	print(numberOfWordsInSentenceBatch)
	
	for l in range(1, numberOfLayers+1):
		Z[generateParameterName(l, "Z")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), dtype=tf.bool), dtype=tf.bool)	#A=Z for binary activations
		A[generateParameterName(l, "A")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), dtype=tf.bool), dtype=tf.bool)	#A=Z for binary activations
		TMax[generateParameterName(l, "TMax")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.int32))
		TMin[generateParameterName(l, "TMin")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.int32))
		for s in range(numberOfSequentialInputs):
			Vseq[generateParameterNameSeq(l, s, "Vseq")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), tf.bool), dtype=tf.bool)
			TMaxSeq[generateParameterNameSeq(l, s, "TMaxSeq")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.int32))
			TMinSeq[generateParameterNameSeq(l, s, "TMinSeq")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.int32))
			Zseq[generateParameterNameSeq(l, s, "Zseq")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), dtype=tf.bool), dtype=tf.bool)	#A=Z for binary activations
			Aseq[generateParameterNameSeq(l, s, "Aseq")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), dtype=tf.bool), dtype=tf.bool)	#A=Z for binary activations
			if(recordSubInputsWeighted):
				numberSubinputsPerSequentialInput = calculateNumberSubinputsPerSequentialInput(s)
				AseqInputVerified[generateParameterNameSeq(l, s, "AseqInputVerified")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, numberSubinputsPerSequentialInput, n_h[l]]), dtype=tf.bool), dtype=tf.bool)
			
			
	for w in range(maxNumberOfWordsInSentenceBatch-numberOfWordsInConvolutionalWindowSeen+1):
	
		#print("w = " + str(w))
		
		for l in range(1, numberOfLayers+1):
			sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), tf.bool), dtype=tf.bool)

		if(w == 0):
			AfirstLayerShifted =  tf.dtypes.cast(x[:, 0:inputLength], tf.bool)
		else:
			paddings = tf.constant([[0, 0], [w*numberOfFeaturesPerWord, 0]])	#shift input to the right by x words (such that a different input window will be presented to the network)
			#AfirstLayerShifted = x[:, w*numberOfFeaturesPerWord:min(w*numberOfFeaturesPerWord+inputLength, numberOfFeaturesCropped)]
			AfirstLayerShifted = tf.dtypes.cast(x[:, w*numberOfFeaturesPerWord:w*numberOfFeaturesPerWord+inputLength], tf.bool)
			tf.pad(AfirstLayerShifted, paddings, "CONSTANT")
		
		#printShape(AfirstLayerShifted, "AfirstLayerShifted")	
		#printAverage(AfirstLayerShifted, "AfirstLayerShifted", 1)
		
		AprevLayer = AfirstLayerShifted
		
		TMaxPrevLayer = tf.ones(([batchSize, n_h[0]]), tf.int32)
		TMaxPrevLayer = TMaxPrevLayer*w
		TMinPrevLayer = tf.ones(([batchSize, n_h[0]]), tf.int32)
		TMinPrevLayer = TMinPrevLayer*w
		
		printAverage(AprevLayer, "AprevLayer", 1)
		printAverage(TMaxPrevLayer, "TMaxPrevLayer", 1)
		printAverage(TMinPrevLayer, "TMinPrevLayer", 1)
		
		for l in range(1, numberOfLayers+1):	#start algorithm at n_h[1]; ie first hidden layer

			#print("\tl = " + str(l))

			#declare variables used across all sequential input of neuron
			#primary vars;
			if(l == 1):
				if(supportSkipLayers):
					AprevLayerAll = AprevLayer	#x
					TMaxPrevLayerAll = TMaxPrevLayer
					TMinPrevLayerAll = TMinPrevLayer
					
					#print(TMinPrevLayer)
					#print(TMaxPrevLayer)
					#print(TMinPrevLayerAll)
					#print(TMaxPrevLayerAll)
					
			else:
				if(supportSkipLayers):
					AprevLayerAll = tf.concat([AprevLayerAll, AprevLayer], 1)
					TMaxPrevLayerAll = tf.concat([TMaxPrevLayerAll, TMaxPrevLayer], 1)
					TMinPrevLayerAll = tf.concat([TMinPrevLayerAll, TMinPrevLayer], 1)

			for sForward in range(numberOfSequentialInputs):
				sReverse = numberOfSequentialInputs-sForward-1	
				s = sReverse
				
				#for each sequential input of each neuron (stating from last), see if its requirements are satisfied
					#if first sequential input, and hypothetical valid activation of first input, allow reset of neuron sequential input 
						#ie if((s == 0) && (resetSequentialInputsIfOnlyFirstInputValid)):
							#skip sequential validation requirements as neuron sequential input can be reset
				
				#print("\t\ts = " + str(s))

				
				#identify (hypothetical) activation of neuron sequential input
				if(supportSkipLayers):
					
					CseqCrossLayerBase = tf.gather(n_h_cumulative['n_h_cumulative'], CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")])
					CseqCrossLayer = tf.add(Cseq[generateParameterNameSeq(l, s, "Cseq")], CseqCrossLayerBase)
					
					AseqInput = tf.gather(AprevLayerAll, CseqCrossLayer, axis=1)
					
					TMaxSeqInput = tf.gather(TMaxPrevLayerAll, CseqCrossLayer, axis=1)
					TMinSeqInput = tf.gather(TMinPrevLayerAll, CseqCrossLayer, axis=1)
					

				else:
					#printAverage(AprevLayer, "AprevLayer", 3)
					#printAverage(TprevLayer, "TprevLayer", 3)
					
					AseqInput = tf.gather(AprevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
					TMaxSeqInput = tf.gather(TMaxPrevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
					TMinSeqInput = tf.gather(TMinPrevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)

				
				#printAverage(AseqInput, "AseqInput", 3)
				#printAverage(TMaxSeqInput, "TMaxSeqInput", 3)
				#printAverage(TMinSeqInput, "TMinSeqInput", 3)
				
				#calculate validation matrix based upon sequentiality requirements
				#if Vseq[s-1] is True and Vseq[s] is False;
					#OLD: and Tseq[s-1] < w [this condition is guaranteed by processing s in reverse]
				if(s > 0):
					VseqTemp = Vseq[generateParameterNameSeq(l, s-1, "Vseq")]
					VseqTemp = tf.math.logical_and(VseqTemp, tf.math.logical_not(Vseq[generateParameterNameSeq(l, s, "Vseq")]))
				else:
					if(resetSequentialInputsIfOnlyFirstInputValid):
						VseqTemp = tf.math.logical_not(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")])
					else:
						VseqTemp = tf.math.logical_and(VseqTemp, tf.math.logical_not(Vseq[generateParameterNameSeq(l, s, "Vseq")]))

				VseqBool = VseqTemp
			
				#printAverage(VseqBool, "VseqBool", 3)
			
				#AseqInput = tf.multiply(VseqBool, AseqInput)
			
				#calculate output for layer sequential input s

				#ensure that T continguous constraint is met;
				#NO: take the max subinput pathway only (ie matrix mult but with max() rather than sum() for each dot product)
				if(s > 0):
					numberSubinputsPerSequentialInput = calculateNumberSubinputsPerSequentialInput(s)
					multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)

					if(enforceTcontiguityBetweenSequentialInputs):
						#1. first select the AseqInput inputs that have TMinSeq[l, s] contiguous with TMaxSeq[l, s-1]:
						TMaxSeqPrevPlus1 = TMaxSeq[generateParameterNameSeq(l, s-1, "TMaxSeq")]+1
						TMinSeqPrev = TMinSeq[generateParameterNameSeq(l, s-1, "TMinSeq")]

						#printAverage(TMaxSeqPrevPlus1, "TMaxSeqPrevPlus1PRE", 3)	
						TMaxSeqPrevPlus1Tiled = tf.tile(tf.reshape(TMaxSeqPrevPlus1, [batchSize, 1, n_h[l]]), multiples)
						#printAverage(TMaxSeqPrevPlus1Tiled, "TMaxSeqPrevPlus1TiledPRE", 3)
						TMinSeqInputThreshold = tf.math.equal(TMinSeqInput, TMaxSeqPrevPlus1Tiled)
						#printAverage(TMinSeqInputThreshold, "TMinSeqInputThresholdPRE", 3)	
						AseqInputTthresholded = tf.math.logical_and(AseqInput, TMinSeqInputThreshold)
						
						TMinSeqInputThresholded = TMinSeqPrev	#note this is recording the min of the sequence, not the sequential input	#OLD: TMaxSeqPrevPlus1
						
						#printAverage(AseqInputTthresholded, "AseqInputTthresholdedPRE", 3)			

						AseqInput = AseqInputTthresholded
						
					if(enforceTcontiguityTakeEncapsulatedSubinputWithMaxTvalueEqualsW):
						TMaxSeqInputThreshold = tf.math.equal(TMaxSeqInput, w)	#CHECKTHIS
						AseqInputTthresholded = tf.math.logical_and(AseqInput, TMaxSeqInputThreshold)

						TMaxSeqInputThresholded = tf.math.reduce_max(TMaxSeqInput, axis=1)	#CHECKTHIS - ensure equal to w
						
						AseqInput = AseqInputTthresholded
						
				else:
					if(resetSequentialInputsIfOnlyFirstInputValid):
						#only reset first sequential input if TMaxSeqInput > TMax[l]

						numberSubinputsPerSequentialInput = calculateNumberSubinputsPerSequentialInput(s)
						multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)
						
						TMaxtiled = tf.tile(tf.reshape(TMax[generateParameterName(l, "TMax")], [batchSize, 1, n_h[l]]), multiples)
						VseqTiled = tf.tile(tf.reshape(Vseq[generateParameterNameSeq(l, s, "Vseq")], [batchSize, 1, n_h[l]]), multiples)

						ZseqHypotheticalResetThreshold = tf.math.logical_or(tf.math.logical_not(VseqTiled), tf.math.greater(TMaxSeqInput, tf.dtypes.cast(TMaxtiled, tf.int32)))
						AseqInputTthresholded = tf.math.logical_and(AseqInput, ZseqHypotheticalResetThreshold)
						
						AseqInput = AseqInputTthresholded
						
						if(enforceTcontiguityTakeEncapsulatedSubinputWithMinTvalue):
							TMinSeqInputThresholdIndices = tf.dtypes.cast(tf.math.argmin(TMinSeqInput, axis=1), tf.int32)
							
							#AseqInput shape: batchSize*numSubinputs*numNeuronsOnLayer
							AseqInputReordedAxes = tf.transpose(AseqInput, [0, 2, 1])
							AseqInputReordedAxesFlattened = tf.reshape(AseqInputReordedAxes, [AseqInputReordedAxes.shape[0]*AseqInputReordedAxes.shape[1], AseqInputReordedAxes.shape[2]])
							idx_0 = tf.reshape(tf.range(AseqInputReordedAxesFlattened.shape[0]), TMinSeqInputThresholdIndices.shape)
							indices=tf.stack([idx_0,TMinSeqInputThresholdIndices],axis=-1)
							AseqInputTthresholded = tf.gather_nd(AseqInputReordedAxesFlattened, indices)
							AseqInputTthresholded = tf.expand_dims(AseqInputTthresholded, axis=1)	#note this dimension will be reduced/eliminated below so its size does not matter
							
							TMinSeqInputThresholded = tf.math.reduce_min(TMinSeqInput, axis=1)
						
							AseqInput = AseqInputTthresholded
							
						if(enforceTcontiguityTakeEncapsulatedSubinputWithMaxTvalueEqualsW):
							TMaxSeqInputThreshold = tf.math.equal(TMaxSeqInput, w)	#CHECKTHIS
							AseqInputTthresholded = tf.math.logical_and(AseqInput, TMaxSeqInputThreshold)

							TMaxSeqInputThresholded = tf.math.reduce_max(TMaxSeqInput, axis=1)	#CHECKTHIS - ensure equal to w
													
							AseqInput = AseqInputTthresholded
				
				
				
				#printAverage(AseqInputTthresholded, "AseqInputTthresholded", 3)			
				#printAverage(TMinSeqInputThresholded, "TMinSeqInputThresholded", 3)
				#printAverage(TMaxSeqInputThresholded, "TMaxSeqInputThresholded", 3)

				
				#take any active and Tthreshold valid sub input:
				ZseqHypothetical = tf.math.reduce_any(AseqInput, axis=1)
			
				#printAverage(ZseqHypothetical, "ZseqHypothetical", 3)
								
				#printAverage(VseqBool, "VseqBool", 3)			
				#printAverage(ZseqHypothetical, "ZseqHypothetical", 3)	
				
				#apply sequential validation matrix
				ZseqHypothetical = tf.math.logical_and(VseqBool, ZseqHypothetical)
				
		

				#printAverage(ZseqHypothetical, "ZseqHypothetical (2)", 3)
				
				#check output threshold
				ZseqPassThresold = ZseqHypothetical	#redundant
				
				ZseqPassThresoldInt = tf.dtypes.cast(ZseqPassThresold, tf.int32)
				ZseqPassThresoldNot = tf.math.logical_not(ZseqPassThresold)
				ZseqPassThresoldNotInt = tf.dtypes.cast(ZseqPassThresoldNot, tf.int32)
				
				if(enforceTcontiguityConstraints):
					TMaxSeqPassThresoldUpdatedValues = tf.multiply(ZseqPassThresoldInt, TMaxSeqInputThresholded)
					TMaxSeqUpdated = tf.multiply(TMaxSeq[generateParameterNameSeq(l, s, "TMaxSeq")], ZseqPassThresoldNotInt)
					TMaxSeqUpdated = tf.add(TMaxSeqUpdated, TMaxSeqPassThresoldUpdatedValues)

					TMinSeqPassThresoldUpdatedValues = tf.multiply(ZseqPassThresoldInt, TMinSeqInputThresholded)
					TMinSeqUpdated = tf.multiply(TMinSeq[generateParameterNameSeq(l, s, "TMinSeq")], ZseqPassThresoldNotInt)
					TMinSeqUpdated = tf.add(TMinSeqUpdated, TMinSeqPassThresoldUpdatedValues)
				
				
				#calculate updated validation matrix
				VseqUpdated = tf.math.logical_or(Vseq[generateParameterNameSeq(l, s, "Vseq")], ZseqPassThresold)
				
				#reset appropriate neurons
				if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
					resetRequiredMatrix = tf.math.logical_and(tf.math.logical_and(ZseqPassThresold, Vseq[generateParameterNameSeq(l, s, "Vseq")]), tf.math.logical_not(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")]))
					for s2 in range(numberOfSequentialInputs):
						Vseq[generateParameterNameSeq(l, s2, "Vseq")] = tf.math.logical_and(Vseq[generateParameterNameSeq(l, s2, "Vseq")], tf.math.logical_not(resetRequiredMatrix))
						if(recordSubInputsWeighted):
							multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)
							resetRequiredMatrixTiled = tf.tile(tf.reshape(resetRequiredMatrix, [batchSize, 1, n_h[l]]), multiples)
							AseqInputVerified[generateParameterNameSeq(l, s2, "AseqInputVerified")] = tf.math.logical_and(AseqInputVerified[generateParameterNameSeq(l, s2, "AseqInputVerified")], tf.math.logical_not(resetRequiredMatrixTiled))
						if(not doNotResetNeuronOutputUntilAllSequentialInputsActivated):
							TMaxSeq[generateParameterNameSeq(l, s2, "TMaxSeq")] = tf.multiply(TMaxSeq[generateParameterNameSeq(l, s2, "TMaxSeq")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))
							TMinSeq[generateParameterNameSeq(l, s2, "TMinSeq")] = tf.multiply(TMinSeq[generateParameterNameSeq(l, s2, "TMinSeq")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))
							Zseq[generateParameterNameSeq(l, s2, "Zseq")] = tf.math.logical_and(Zseq[generateParameterNameSeq(l, s2, "Zseq")], tf.math.logical_not(resetRequiredMatrix))
							Aseq[generateParameterNameSeq(l, s2, "Aseq")] = tf.math.logical_and(Aseq[generateParameterNameSeq(l, s2, "Aseq")], tf.math.logical_not(resetRequiredMatrix))
					sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.math.logical_and(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")], tf.math.logical_not(resetRequiredMatrix))
					if(not doNotResetNeuronOutputUntilAllSequentialInputsActivated):
						Z[generateParameterName(l, "Z")] = tf.math.logical_and(Z[generateParameterName(l, "Z")], tf.math.logical_not(resetRequiredMatrix))
						A[generateParameterName(l, "A")] = tf.math.logical_and(A[generateParameterName(l, "A")], tf.math.logical_not(resetRequiredMatrix))
						TMax[generateParameterName(l, "TMax")] = tf.multiply(TMax[generateParameterName(l, "TMax")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))
						TMin[generateParameterName(l, "TMin")] = tf.multiply(TMin[generateParameterName(l, "TMin")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))

				#store updated validation matrix
				Vseq[generateParameterNameSeq(l, s, "Vseq")] = VseqUpdated
				if(enforceTcontiguityConstraints):
					TMaxSeq[generateParameterNameSeq(l, s, "TMaxSeq")] = TMaxSeqUpdated
					TMinSeq[generateParameterNameSeq(l, s, "TMinSeq")] = TMinSeqUpdated
				sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.math.logical_or(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")], ZseqPassThresold)
				
				
				#calculate thresholded output
				ZseqThresholded = ZseqPassThresold	#redundant

				#update Zseq
				ZseqUpdated = tf.math.logical_or(Zseq[generateParameterNameSeq(l, s, "Zseq")], ZseqPassThresold) 
				Zseq[generateParameterNameSeq(l, s, "Zseq")] = ZseqUpdated
			
				#printAverage(ZseqPassThresold, "ZseqPassThresold", 3)
				#printAverage(VseqUpdated, "VseqUpdated", 3)
				#printAverage(ZseqUpdated, "ZseqUpdated", 3)	
				#printAverage(TMaxSeqUpdated, "TMaxSeqUpdated", 3)
				#printAverage(TMinSeqUpdated, "TMinSeqUpdated", 3)
				
				#regenerate Aseq after Zseq update
				Aseq[generateParameterNameSeq(l, s, "Aseq")] = Zseq[generateParameterNameSeq(l, s, "Zseq")]
				
				#printAverage(Zseq[generateParameterNameSeq(l, s, "Zseq")], "Zseq", 3)
				
				if(recordSubInputsWeighted):
					#apply sequential verification matrix to AseqInput (rather than ZseqHypothetical) to record the precise subinputs that resulted in a sequential input being activated:
					
					numberSubinputsPerSequentialInput = calculateNumberSubinputsPerSequentialInput(s)
					multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)
					
					VseqBoolTiled =  tf.tile(tf.reshape(VseqBool, [batchSize, 1, n_h[l]]), multiples)
					AseqInputVerifiedTemp = tf.math.logical_and(AseqInput, VseqBoolTiled)
											
					AseqInputVerified[generateParameterNameSeq(l, s, "AseqInputVerified")] = tf.math.logical_or(AseqInputVerified[generateParameterNameSeq(l, s, "AseqInputVerified")], AseqInputVerifiedTemp)			
				
				if(s == numberOfSequentialInputs-1):
					ZseqLast = Zseq[generateParameterNameSeq(l, s, "Zseq")]
					AseqLast = Aseq[generateParameterNameSeq(l, s, "Aseq")]
					VseqLast = Vseq[generateParameterNameSeq(l, s, "Vseq")]
					TMaxSeqLast = TMaxSeq[generateParameterNameSeq(l, s, "TMaxSeq")]
					TMinSeqLast = TMinSeq[generateParameterNameSeq(l, s, "TMinSeq")]
				
					#printAverage(TMaxSeqLast, "TMaxSeqLast", 3)
					#printAverage(TMinSeqLast, "TMinSeqLast", 3)
				
					if(recordNetworkWeights):
						if(recordSubInputsWeighted):
							for s2 in range(numberOfSequentialInputs):
								numberSubinputsPerSequentialInput = calculateNumberSubinputsPerSequentialInput(s2)
								multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)
								neuronNewlyActivatedTiled = tf.tile(tf.reshape(ZseqPassThresold, [batchSize, 1, n_h[l]]), multiples)	#or; VseqBool (since will be logically anded with AseqInputVerified)

								AseqInputVerifiedAndNeuronActivated = tf.math.logical_and(AseqInputVerified[generateParameterNameSeq(l, s2, "AseqInputVerified")], neuronNewlyActivatedTiled)
								AseqInputVerifiedAndNeuronActivatedBatchSummed = tf.math.reduce_sum(tf.dtypes.cast(AseqInputVerifiedAndNeuronActivated, tf.float32), axis=0)

								if(recordSequentialInputsWeighted):
									WRseq[generateParameterNameSeq(l, s2, "WRseq")] = tf.add(WRseq[generateParameterNameSeq(l, s2, "WRseq")], AseqInputVerifiedAndNeuronActivatedBatchSummed)

							if(recordNeuronsWeighted):
								neuronNewlyActivated = ZseqPassThresold
								neuronNewlyActivatedBatchSummed = tf.math.reduce_sum(tf.dtypes.cast(neuronNewlyActivated, tf.float32), axis=0)

								if(recordNeuronsWeighted):
									BR[generateParameterName(l, "BR")] = tf.add(BR[generateParameterName(l, "BR")], neuronNewlyActivatedBatchSummed)
					
				if(recordNetworkWeights):	
					if(recordSequentialInputsWeighted):	
						sequentialInputNewlyActivated = ZseqPassThresold
						sequentialInputNewlyActivatedBatchSummed = tf.math.reduce_sum(tf.dtypes.cast(sequentialInputNewlyActivated, tf.float32), axis=0)

						if(recordSequentialInputsWeighted):	
							WR[generateParameterName(l, "WR")] = tf.add(WR[generateParameterName(l, "WR")][s], sequentialInputNewlyActivatedBatchSummed)
				
			Z1 = ZseqLast
			A1 = AseqLast
											
			A[generateParameterName(l, "A")] = A1
			Z[generateParameterName(l, "Z")] = Z1
			
			TMax[generateParameterName(l, "TMax")] = TMaxSeqLast
			TMin[generateParameterName(l, "TMin")] = TMinSeqLast
			
			AprevLayer = A[generateParameterName(l, "A")]
			TMaxPrevLayer = TMax[generateParameterName(l, "TMax")]
			TMinPrevLayer = TMin[generateParameterName(l, "TMin")]
	
	#return tf.nn.softmax(Z1)
	#return tf.nn.sigmoid(Z1)	#binary classification
	
	ZlastLayer = Z[generateParameterName(numberOfLayers, "Z")]
	if(enforceTcontiguityStartAndEndOfSequence):
		TMaxLastLayer = TMax[generateParameterName(numberOfLayers, "TMax")] 
		TMinLastLayer = TMin[generateParameterName(numberOfLayers, "TMin")]
		
		printAverage(ZlastLayer, "ZlastLayer", 1)
		printAverage(TMaxLastLayer, "TMaxLastLayer", 1)
		printAverage(TMinLastLayer, "TMinLastLayer", 1)
		
		#multiples = tf.constant([1,numberOfLayers], tf.int32)
		#numberOfWordsInSentenceBatchTiled = tf.tile(tf.reshape(numberOfWordsInSentenceBatch, [batchSize, n_h[numberOfLayers]]), multiples)
		multiples = tf.constant([1,n_h[numberOfLayers]], tf.int32)
		numberOfWordsInSentenceBatchTiled = tf.tile(tf.reshape(numberOfWordsInSentenceBatch, [batchSize, 1]), multiples)
		
		TMaxLastLayerThreshold = tf.math.equal(TMaxLastLayer, numberOfWordsInSentenceBatchTiled-1)
		printAverage(TMaxLastLayerThreshold, "TMaxLastLayerThreshold", 1)
		TMinLastLayerThreshold = tf.math.equal(TMinLastLayer, 0)
		printAverage(TMaxLastLayerThreshold, "TMaxLastLayerThreshold", 1)
		
		ZlastLayer = tf.math.logical_and(ZlastLayer, TMaxLastLayerThreshold)
		printAverage(ZlastLayer, "ZlastLayer", 1)
		ZlastLayer = tf.math.logical_and(ZlastLayer, TMinLastLayerThreshold)
		printAverage(ZlastLayer, "ZlastLayer", 1)
	
	return tf.math.reduce_any(ZlastLayer, axis=1)

	


