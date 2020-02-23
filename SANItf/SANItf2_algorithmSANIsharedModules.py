# -*- coding: utf-8 -*-
"""SANItf2_algorithmSANIsharedModules.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Usage:
see SANItf2.py

# Description:

Define Sequentially Activated Neuronal Input (SANI) neural net - shared modules 

Neural modules can be shared between different areas of input sequence, eg sentence (cf RNN).

This code mirrors that of GIA Sequence Grammar ANN.
  
Can parse (by default expects to parse) full sentences; ie features for each word in sentence.

- Author: Richard Bruce Baxter - Copyright (c) 2020 Baxter AI (baxterai.com)

"""

import tensorflow as tf
import numpy as np
from SANItf2_operations import * #generateParameterNameSeq, generateParameterName
import SANItf2_globalDefs

numberOfFeaturesPerWord = -1
paddingTagIndex = -1

sequentialInputActivationThreshold = 0.1	#CHECKTHIS (requires optimisation)


allowMultipleSubinputsPerSequentialInput = False

	
if(allowMultipleSubinputsPerSequentialInput):
	allowMultipleContributingSubinputsPerSequentialInput = False	#whether only 1 subinput can be fired to activate a sequential input	
	if(allowMultipleContributingSubinputsPerSequentialInput):
		numberOfWordsInConvolutionalWindowSeen = 10	#the convolutional window (kernel) captures x words every time is slided to the right
	else:
		numberOfWordsInConvolutionalWindowSeen = 1	#CHECKTHIS
else:
	allowMultipleContributingSubinputsPerSequentialInput = False	#mandatory
	numberOfWordsInConvolutionalWindowSeen = 1
	
resetSequentialInputsIfOnlyFirstInputValid = True	#see GIA_TXT_REL_TRANSLATOR_NEURAL_NETWORK_SEQUENCE_GRAMMAR development history for meaning and algorithmic implications of this feature
if(resetSequentialInputsIfOnlyFirstInputValid):
	if(allowMultipleContributingSubinputsPerSequentialInput):
		averageTimeChangeOfNewInputRequiredForReset = 1
	doNotResetNeuronOutputUntilAllSequentialInputsActivated = True

if(allowMultipleSubinputsPerSequentialInput):
	if(allowMultipleContributingSubinputsPerSequentialInput):
		useSparseTensors = False	#optional
	else:
		useSparseTensors = True		#mandatory	#FUTURE: upgrade code to remove this requirement 
else:
	useSparseTensors = True	#mandatory	#sparse tensors are used 

if(useSparseTensors):	#FUTURE: upgrade code to remove this requirement
	if(allowMultipleSubinputsPerSequentialInput):
		#if(numberOfSequentialInputs == 2):
		oneSequentialInputHasOnlyOneSubinput = True	#conditional probability determination of events
		if(oneSequentialInputHasOnlyOneSubinput):
			firstSequentialInputHasOnlyOneSubinput = True #use combination of allowMultipleSubinputsPerSequentialInput for different sequential inputs;  #1[#2] sequential input should allow multiple subinputs, #2[#1] sequential input should allow single subinput
			if(firstSequentialInputHasOnlyOneSubinput):
				lastSequentialInputHasOnlyOneSubinput = False
			else:
				lastSequentialInputHasOnlyOneSubinput = True

if(allowMultipleSubinputsPerSequentialInput):
	if(useSparseTensors):
		supportSkipLayers = True
		if(oneSequentialInputHasOnlyOneSubinput):
			numberSubinputsPerSequentialInput = 50	#~approx equal number of prev layer neurons/2	#number of prior/future events in which to calculate a conditional probability
		else:
			numberSubinputsPerSequentialInput = 3	#sparsity
	else:
		supportSkipLayers = True
else:
	supportSkipLayers = True
	

if(allowMultipleSubinputsPerSequentialInput):
	performSummationOfSequentialInputs = True	#optional
	
	if(allowMultipleContributingSubinputsPerSequentialInput):
		#[multiple contributing subinputs per sequential input] #each sequential input can detect a pattern of activation from the previous layer
		performSummationOfSubInputs = True	#mandatory (implied)
		performSummationOfSubInputsWeighted = True	#mandatory?
		performSummationOfSubInputsNonlinear = True
	else:
		performSummationOfSubInputs = False	#optional though by algorithm design: False
		performSummationOfSubInputsWeighted = False	#will take (True: most weighted) (False: any) active time contiguous subinput
		performSummationOfSubInputsNonlinear = False
		
	if(performSummationOfSequentialInputs):
		performSummationOfSequentialInputsWeighted = False #True	#determines if backprop is required to update weight matrix associated with sequential inputs
		performSummationOfSequentialInputsNonlinear = False	#True	#applies nonlinear function to weighting
		performSummationOfSequentialInputsVerify = True	#verify that all (last) sequential inputs are activated	#CHECKTHIS
	else:
		performSummationOfSequentialInputsWeighted = False
		performSummationOfSequentialInputsNonlinear = False
		performSummationOfSequentialInputsVerify = False
		
	if(performSummationOfSequentialInputs):
		if(performSummationOfSequentialInputsWeighted):
			#sequentialInputCombinationModeSummation = 3
			sequentialInputCombinationModeSummation = 4
		else:
			#sequentialInputCombinationModeSummation = 1
			sequentialInputCombinationModeSummation = 2
			sequentialInputCombinationModeSummationAveraged = True	
	else:
		useLastSequentialInputOnly = True	#implied variable (not used)
		
else:
	performSummationOfSubInputsWeighted = False
	
	performSummationOfSequentialInputs = True

	if(performSummationOfSequentialInputs):
		performSummationOfSequentialInputsWeighted = True	#does backprop require to update weight matrix associated with sequential inputs?
		performSummationOfSequentialInputsNonlinear = True
		performSummationOfSequentialInputsVerify = True	#verify that all (last) sequential inputs are activated	#CHECKTHIS
	else:
		performSummationOfSequentialInputsWeighted = False
		performSummationOfSequentialInputsNonlinear = False
		performSummationOfSequentialInputsVerify = True	#verify that all (last) sequential inputs are activated	#CHECKTHIS
		

	if(performSummationOfSequentialInputs):
		if(performSummationOfSequentialInputsWeighted):
			sequentialInputCombinationModeSummation = 3
		else:
			sequentialInputCombinationModeSummation = 1	
	else:
		useLastSequentialInputOnly = True	#implied variable (not used)
		
			

				
#variable parameters (tf.variable):
if(allowMultipleSubinputsPerSequentialInput):
	if(performSummationOfSubInputsWeighted):
		Wseq = {}	#weights matrix
		Bseq = {}	#biases vector
if(performSummationOfSequentialInputsWeighted):	
	W = {}	#weights matrix
	B = {}	#biases vector
			

#static parameters (convert from tf.variable to tf.constant?):
if(useSparseTensors):
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
Tseq = {}
ZseqTadjusted = {}
Z = {}
A = {}
T = {}
sequentialActivationFound = {}

def defineTrainingParametersSANIsharedModules(numberOfFeaturesPerWordNew, paddingTagIndexNew):
	global numberOfFeaturesPerWord
	global paddingTagIndex
	numberOfFeaturesPerWord = numberOfFeaturesPerWordNew
	paddingTagIndex = paddingTagIndexNew

def defineTrainingParametersSANI(dataset, trainMultipleFiles):
	#Training parameters
	if(trainMultipleFiles):
		learningRate = 0.001
		if(dataset == "POStagSentence"):
			trainingSteps = 10000
	else:
		learningRate = 0.001
		if(dataset == "POStagSentence"):
			trainingSteps = 10000

	if(allowMultipleSubinputsPerSequentialInput):
		batchSize = 100
		displayStep = 100
	else:
		batchSize = 10
		displayStep = 100	

	return learningRate, trainingSteps, batchSize, displayStep
	
def defineNetworkParametersSANI(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles, useSmallSentenceLengths):

	#useSmallSentenceLengths not implemented

	global n_h
	global numberOfLayers
	global numberOfSequentialInputs
	
	inputLength = numberOfFeaturesPerWord*numberOfWordsInConvolutionalWindowSeen

	n_x = inputLength #datasetNumFeatures
	n_y = 1  #SANIshared uses a single output neuron (either 1 or 0)	#if multiple output classes: n_y = num_output_neurons-1 or datasetNumClasses-1	
	n_h_0 = n_x
	if(dataset == "POStagSentence"):
		if(allowMultipleSubinputsPerSequentialInput):
			#n_h_1 = int(inputLength)
			#n_h_2 = int(inputLength/4)
			#n_h_3 = n_y
			#n_h = [n_h_0, n_h_1, n_h_2, n_h_3]
			n_h_1 = int(inputLength)
			n_h_2 = int(inputLength/2)
			n_h_3 = int(inputLength/4)
			n_h_4 = int(inputLength/8)
			n_h_5 = n_y
			n_h = [n_h_0, n_h_1, n_h_2, n_h_3, n_h_4, n_h_5]
		else:
			#FUTURE: the number of neurons/connections should be greatly increased, then pruned
			#n_h_1 = int(datasetNumFeatures*3)
			#n_h_2 = int(datasetNumFeatures/2)
			#n_h_3 = n_y
			#n_h = [n_h_0, n_h_1, n_h_2, n_h_3]
			n_h_1 = int(datasetNumFeatures*1)	#*x for skip layers	#FUTURE: upgrade to support multiple permutations
			n_h_2 = int(datasetNumFeatures*2)
			n_h_3 = int(datasetNumFeatures*3)
			n_h_4 = int(datasetNumFeatures*4)
			n_h_5 = n_y
			n_h = [n_h_0, n_h_1, n_h_2, n_h_3, n_h_4, n_h_5]
	else:
		print("sequential input data is required")
		exit()
		
	numberOfLayers = len(n_h)-1
	numberOfSequentialInputs = 2	#3
	
	
def neuralNetworkPropagationSANI(x):
		
	#note connectivity indexes are used rather than sparse weight matrices due to limitations in current tf2 sparse tensor implementation
	
	#definitions for reference:
	
	#neuron sequential input vars;
	#x/AprevLayer	#output vector (dim: batchSize*n_h[l])
	#if(allowMultipleSubinputsPerSequentialInput):
		#if(useSparseTensors):
			#Cseq	#static connectivity matrix (int) - indexes of neurons on prior layer stored; mapped to W  (dim: numberSubinputsPerSequentialInput*n_h[l])
			#if(supportSkipLayers):
				#CseqLayer	#static connectivity matrix (int) - indexes of pior layer stored; mapped to W (dim: numberSubinputsPerSequentialInput*n_h[l])
			#Wseq #weights of connections; see Cseq (dim: numberSubinputsPerSequentialInput*n_h[l])
			#AseqSum	#combination variable
		#else:
			#if(supportSkipLayers):
				#Wseq #weights of connections (dim: n_h_cumulativeNP[l-1]*n_h[l])
			#else:
				#Wseq #weights of connections (dim: n_h[l-1]*n_h[l])
	#else:
		#Cseq	#static connectivity vector (int) - indexes of neurons on prior layer stored; mapped to W - defines which prior layer neuron a sequential input is connected to (dim: n_h[l])
		#if(supportSkipLayers):
			#CseqLayer	#static connectivity matrix (int) - indexes of pior layer stored; mapped to W - defines which prior layer a sequential input is connected to  (dim: n_h[l])
		#Wseq #weights of connections; see Cseq (dim: n_h[l])	
	
	#Vseq	#mutable verification vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)			#records whether particular neuron sequential inputs are currently active
	#Tseq	#mutable time vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)				#records the time at which a particular sequential input last fired
	#Zseq	#neuron activation function output vector (dim: batchSize*n_h[l]  - regenerated for each sequential input index)	#records the current output value of sequential inputs
	#Aseq	#neuron activation function output vector (dim: batchSize*n_h[l]  - regenerated for each sequential input index)	#records the current output value of sequential inputs
	#if(allowMultipleContributingSubinputsPerSequentialInput):
		#if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
			#ZseqTadjusted	#neuron activation function output vector T adjusted (dim: batchSize*n_h[l]  - regenerated for each sequential input index)	#records the current output value of sequential inputs*time
	
	#Q  
	#Z	#neuron activation function input (dim: batchSize*n_h[l])
	#A	#neuron activation function output (dim: batchSize*n_h[l])
	#T	#mutable time vector (dim: batchSize*n_h[l]) - same as Tseq[numberOfSequentialInputs-1]
	#if(performSummationOfSequentialInputsWeighted):	
		#W	(dim: numberOfSequentialInputs*n_h[l])	
	
	#combination vars (per layer)???;
	#if(performSummationOfSequentialInputs):
		#these are all used for different methods of sequential input summation:
		#if(sequentialInputCombinationModeSummation == 1):
			#ZseqSum	#(dim: batchSize*n_h[l])
		#if(sequentialInputCombinationModeSummation == 2):
			#AseqSum	#(dim: batchSize*n_h[l])
		#if(sequentialInputCombinationModeSummation == 3):
			#ZseqWeightedSum	#(dim: batchSize*n_h[l])
		#if(sequentialInputCombinationModeSummation == 4):
			#AseqWeightedSum	#(dim: batchSize*n_h[l])
		
	batchSize = x.shape[0]
	
	#optimise feed length based on max sentence length in batch:
	#unoptimised: numberOfFeatures = x.shape[1]
	#print(x)
	xIsNotPadding = tf.math.less(x, paddingTagIndex) #tf.math.less(tf.dtypes.cast(x, tf.int32), paddingTagIndex)
	#print(xIsNotPadding)
	coordinatesOfNotPadding = tf.where(xIsNotPadding)
	#print(coordinatesOfNotPadding)
	numberOfFeaturesCropped = tf.reduce_max(coordinatesOfNotPadding[:, 1]).numpy()+1
	#print("numberOfFeaturesCropped = ", numberOfFeaturesCropped)
	
	inputLength = numberOfFeaturesPerWord*numberOfWordsInConvolutionalWindowSeen
	
	#print("numberOfFeaturesPerWord = ", numberOfFeaturesPerWord)
	#print("numberOfWordsInConvolutionalWindowSeen = ", numberOfWordsInConvolutionalWindowSeen)
	#print("inputLength = ", inputLength)
	
	maxNumberOfWordsInSentenceBatch = int(numberOfFeaturesCropped/numberOfFeaturesPerWord)
	
	for l in range(1, numberOfLayers+1):
		Z[generateParameterName(l, "Z")] = tf.Variable(tf.zeros([batchSize, n_h[l]]), dtype=tf.float32)
		A[generateParameterName(l, "A")] = tf.Variable(tf.zeros([batchSize, n_h[l]]), dtype=tf.float32)
		T[generateParameterName(l, "T")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.int32))
		for s in range(numberOfSequentialInputs):
			Vseq[generateParameterNameSeq(l, s, "Vseq")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), tf.bool), dtype=tf.bool)
			Tseq[generateParameterNameSeq(l, s, "Tseq")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.int32))
			Zseq[generateParameterNameSeq(l, s, "Zseq")] = tf.Variable(tf.zeros([batchSize, n_h[l]]), dtype=tf.float32)
			Aseq[generateParameterNameSeq(l, s, "Aseq")] = tf.Variable(tf.zeros([batchSize, n_h[l]]), dtype=tf.float32)
			if(allowMultipleContributingSubinputsPerSequentialInput):
				if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
					ZseqTadjusted[generateParameterNameSeq(l, s, "ZseqTadjusted")] = tf.Variable(tf.zeros([batchSize, n_h[l]]), dtype=tf.float32)

	#for w in range(maxNumberOfWordsInSentenceBatch):
	#for w in range(0, 1):
	for w in range(maxNumberOfWordsInSentenceBatch-numberOfWordsInConvolutionalWindowSeen+1):
	
		#print("w = " + str(w))
		
		for l in range(1, numberOfLayers+1):
			sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), tf.bool), dtype=tf.bool)

		if(w == 0):
			AfirstLayerShifted = x[:, 0:inputLength]
		else:
			paddings = tf.constant([[0, 0], [w*numberOfFeaturesPerWord, 0]])	#shift input to the right by x words (such that a different input window will be presented to the network)
			#AfirstLayerShifted = x[:, w*numberOfFeaturesPerWord:min(w*numberOfFeaturesPerWord+inputLength, numberOfFeaturesCropped)]
			AfirstLayerShifted = x[:, w*numberOfFeaturesPerWord:w*numberOfFeaturesPerWord+inputLength]
			tf.pad(AfirstLayerShifted, paddings, "CONSTANT")
		
		#printShape(AfirstLayerShifted, "AfirstLayerShifted")
			
		AprevLayer = AfirstLayerShifted
		
		TprevLayer = tf.ones(([batchSize, n_h[0]]), tf.int32)
		TprevLayer = TprevLayer*w
		
		for l in range(1, numberOfLayers+1):	#start algorithm at n_h[1]; ie first hidden layer

			#print("\tl = " + str(l))

			#declare variables used across all sequential input of neuron
			#primary vars;
			if(l == 1):
				if(supportSkipLayers):
					AprevLayerAll = AprevLayer	#x
					TprevLayerAll = TprevLayer
			else:
				if(supportSkipLayers):
					AprevLayerAll = tf.concat([AprevLayerAll, AprevLayer], 1)
					TprevLayerAll = tf.concat([TprevLayerAll, TprevLayer], 1)

			#combination vars;
			if(performSummationOfSequentialInputs):
				#these are all used for different methods of sequential input summation:
				if(sequentialInputCombinationModeSummation == 1):
					ZseqSum = tf.zeros([batchSize, n_h[l]], tf.float32)
				if(sequentialInputCombinationModeSummation == 2):
					AseqSum = tf.zeros([batchSize, n_h[l]], tf.float32)
				if(sequentialInputCombinationModeSummation == 3):
					ZseqWeightedSum = tf.zeros([batchSize, n_h[l]], tf.float32)
				if(sequentialInputCombinationModeSummation == 4):
					AseqWeightedSum = tf.zeros([batchSize, n_h[l]], tf.float32)

			
			for sForward in range(numberOfSequentialInputs):
				sReverse = numberOfSequentialInputs-sForward-1	
				s = sReverse
				
				#for each sequential input of each neuron (stating from last), see if its requirements are satisfied
					#if first sequential input, and hypothetical valid activation of first input, allow reset of neuron sequential input 
						#ie if((s == 0) && (resetSequentialInputsIfOnlyFirstInputValid)):
							#skip sequential validation requirements as neuron sequential input can be reset
				
				#print("\t\ts = " + str(s))

				#identify (hypothetical) activation of neuron sequential input
				if(useSparseTensors):
					if(supportSkipLayers):
					
						if(l == 1):
							CseqCrossLayerBase = 0
						else:
							CseqCrossLayerBase = tf.gather(n_h_cumulative['n_h_cumulative'], CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")])
						CseqCrossLayer = tf.add(Cseq[generateParameterNameSeq(l, s, "Cseq")], CseqCrossLayerBase)
						#printShape(AprevLayerAll, "AprevLayerAll")
						#printShape(CseqCrossLayer, "CseqCrossLayer")						
						AseqInput = tf.gather(AprevLayerAll, CseqCrossLayer, axis=1)
						if(allowMultipleContributingSubinputsPerSequentialInput):
							if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
								AseqInputTadjusted = tf.gather(tf.multiply(AprevLayerAll, tf.dtypes.cast(TprevLayerAll, tf.float32)), CseqCrossLayer, axis=1) 
						if(not allowMultipleContributingSubinputsPerSequentialInput):
							#printShape(TprevLayerAll, "TprevLayerAll")
							#printShape(CseqCrossLayer, "CseqCrossLayer")
							TseqInput = tf.gather(TprevLayerAll, CseqCrossLayer, axis=1)
					else:
						AseqInput = tf.gather(AprevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
						if(allowMultipleContributingSubinputsPerSequentialInput):
							if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
								AseqInputTadjusted = tf.gather(tf.multiply(AprevLayer, tf.dtypes.cast(TprevLayer, tf.float32)), Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
						if(not allowMultipleContributingSubinputsPerSequentialInput):
							TseqInput = tf.gather(TprevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
				else:
					if(supportSkipLayers):
						AseqInput = AprevLayerAll
						if(allowMultipleContributingSubinputsPerSequentialInput):
							if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
								AseqInputTadjusted = tf.multiply(AprevLayerAll, tf.dtypes.cast(TprevLayerAll, tf.float32))
						if(not allowMultipleContributingSubinputsPerSequentialInput):
							TseqInput = TprevLayerAll
					else:
						AseqInput = AprevLayer
						if(allowMultipleContributingSubinputsPerSequentialInput):
							if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
								AseqInputTadjusted = tf.multiply(AprevLayer, tf.dtypes.cast(TprevLayer, tf.float32))
						if(not allowMultipleContributingSubinputsPerSequentialInput):
							TseqInput = TprevLayer
		
									
				#calculate validation matrix based upon sequentiality requirements
				#if Vseq[s-1] is True and Vseq[s] is False;
					#OLD: and Tseq[s-1] < w [this condition is guaranteed by processing s in reverse]
				if(s > 0):
					VseqTemp = Vseq[generateParameterNameSeq(l, s-1, "Vseq")]
					#VseqTemp = tf.math.logical_and(VseqTemp, tf.math.logical_less(Tseq[generateParameterNameSeq(l, s-1, "Tseq")], w))
					VseqTemp = tf.math.logical_and(VseqTemp, tf.math.logical_not(Vseq[generateParameterNameSeq(l, s, "Vseq")]))
				else:
					if(resetSequentialInputsIfOnlyFirstInputValid):
						#VseqTemp = tf.math.logical_or(tf.math.logical_not(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")]), tf.math.logical_not(Vseq[generateParameterNameSeq(l, s, "Vseq")]))
						VseqTemp = tf.math.logical_not(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")])
					else:
						VseqTemp = tf.math.logical_and(VseqTemp, tf.math.logical_not(Vseq[generateParameterNameSeq(l, s, "Vseq")]))

				#VseqInt = tf.dtypes.cast(VseqTemp, tf.int32)
				VseqFloat = tf.dtypes.cast(VseqTemp, tf.float32)
				
				#AseqInput = tf.multiply(VseqFloat, AseqInput)
			
				#calculate output for layer sequential input s
				if(allowMultipleSubinputsPerSequentialInput):
					if(not useSparseTensors):
						if(performSummationOfSubInputsWeighted):
							ZseqHypothetical = tf.add(tf.matmul(AseqInput, Wseq[generateParameterNameSeq(l, s, "Wseq")]), Bseq[generateParameterNameSeq(l, s, "Bseq")])
							#ZseqHypothetical = tf.matmul(AseqInput, Wseq[generateParameterNameSeq(l, s, "Wseq")])
						else:
							AseqInputAverage = tf.math.reduce_mean(AseqInput, axis=1)	#take average
							multiples = tf.constant([1,n_h[l]], tf.int32)
							ZseqHypothetical = tf.tile(tf.reshape(AseqInputAverage, [batchSize, 1]), multiples)
					else:
						if(not allowMultipleContributingSubinputsPerSequentialInput):
							#ensure that T continguous constraint is met;
							#NO: take the max subinput pathway only (ie matrix mult but with max() rather than sum() for each dot product)
							if(s > 0):
								#1. first select the AseqInput inputs that have T[l-1] contiguous with Tseq[l, s-1]:
								TseqPlus1 = Tseq[generateParameterNameSeq(l, s-1, "Tseq")]+1

								multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)

								TseqPlus1Tiled = tf.tile(tf.reshape(TseqPlus1, [batchSize, 1, n_h[l]]), multiples)
								TseqInputThreshold = tf.math.equal(TseqInput, TseqPlus1Tiled)

								AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(TseqInputThreshold, tf.float32))
								AseqInput = AseqInputTthresholded

								#VseqFloatTiled = tf.tile(tf.reshape(VseqFloat, [batchSize, 1, n_h[l]]), multiples)
								#AseqInputThresholded = tf.multiply(VseqFloatTiled, AseqInput)	#done later
							else:
								if(resetSequentialInputsIfOnlyFirstInputValid):
									#NO: ZseqHypotheticalResetThreshold = 1 	#always reset if valid first input (CHECKTHIS)	#dummy variable
									#only reset first sequential input if TseqInput > T[l]

									Ttiled = tf.tile(tf.reshape(T[generateParameterName(l, "T")], [batchSize, 1, n_h[l]]), multiples)
									VseqTiled = tf.tile(tf.reshape(Vseq[generateParameterNameSeq(l, s, "Vseq")], [batchSize, 1, n_h[l]]), multiples)

									ZseqHypotheticalResetThreshold = tf.math.logical_or(tf.math.logical_not(VseqTiled), tf.math.greater(TseqInput, tf.dtypes.cast(Ttiled, tf.int32)))

									AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(ZseqHypotheticalResetThreshold, tf.float32))
									AseqInput = AseqInputTthresholded


						#2. take the AseqInput with the highest weighting
						if(performSummationOfSubInputsWeighted):
							multiplesSeq = tf.constant([batchSize,1,1], tf.int32)
							WseqTiled = tf.tile(tf.reshape(Wseq[generateParameterNameSeq(l, s, "Wseq")], [1, numberSubinputsPerSequentialInput, n_h[l]]), multiplesSeq)
							AseqInputWeighted = tf.multiply(AseqInput, WseqTiled)
						else:
							AseqInputWeighted = AseqInput

						if(performSummationOfSubInputs):
							ZseqHypothetical = tf.math.reduce_sum(AseqInputWeighted, axis=1)
						else:
							#take sub input with max input signal*weight
							ZseqHypothetical = tf.math.reduce_max(AseqInputWeighted, axis=1)
							ZseqHypotheticalIndex = tf.math.argmax(AseqInputWeighted, axis=1)
				else:
					#ensure that T continguous constraint is met;
					if(s > 0):
						#1. first select the AseqInput inputs that have T[l-1] contiguous with Tseq[l, s-1]:
						
						TseqPlus1 = Tseq[generateParameterNameSeq(l, s-1, "Tseq")]+1
						TseqInputThreshold = tf.math.equal(TseqInput, TseqPlus1)

						AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(TseqInputThreshold, tf.float32))
						AseqInput = AseqInputTthresholded
					else:
						#only reset first sequential input if TseqInput > T[l]

						ZseqHypotheticalResetThreshold = tf.math.logical_or(tf.math.logical_not(Vseq[generateParameterNameSeq(l, s, "Vseq")]), tf.math.greater(TseqInput, tf.dtypes.cast(T[generateParameterName(l, "T")], tf.int32)))

						AseqInputTthresholded = tf.multiply(AseqInput, tf.dtypes.cast(ZseqHypotheticalResetThreshold, tf.float32))
						AseqInput = AseqInputTthresholded
					
					ZseqHypothetical = AseqInput	#CHECKTHIS
					
					

				#apply sequential validation matrix
				ZseqHypothetical = tf.multiply(VseqFloat, ZseqHypothetical)
				
				#check output threshold
				ZseqPassThresold = tf.math.greater(ZseqHypothetical, sequentialInputActivationThreshold)
				
				#update ZseqPassThresold based on reset
				if(allowMultipleSubinputsPerSequentialInput):
					if(allowMultipleContributingSubinputsPerSequentialInput):
						if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
							#ensure that T continguous constraint is met;
							
							#calculate output for layer sequential input s
							#old slow: only factor in inputs that have changed since the component was originally activated; if Torigin > Tseq[s]
							#new: only factor in inputs if ZseqHypotheticalTadjusted > ZseqTadjusted+1
							ZseqHypotheticalTadjusted = tf.add(tf.matmul(AseqInputTadjusted, Wseq[generateParameterNameSeq(l, s, "Wseq")]), Bseq[generateParameterNameSeq(l, s, "Bseq")])
							ZseqHypotheticalTadjusted = tf.divide(ZseqHypotheticalTadjusted, AseqInputTadjusted.shape[1])	#normalise T adjustment
							ZseqHypotheticalTadjustedResetThreshold = tf.math.logical_or(tf.math.logical_not(Vseq[generateParameterNameSeq(l, s, "Vseq")]), tf.math.greater(ZseqHypotheticalTadjusted, ZseqTadjusted[generateParameterNameSeq(l, s, "ZseqTadjusted")]+averageTimeChangeOfNewInputRequiredForReset))

							ZseqPassThresold = tf.math.logical_and(ZseqPassThresold, ZseqHypotheticalTadjustedResetThreshold)	#ensures that if reset is required, Tadjusted threshold is met
				
				ZseqPassThresoldInt = tf.dtypes.cast(ZseqPassThresold, tf.int32)
				TseqPassThresoldUpdatedValues = tf.multiply(ZseqPassThresoldInt, w)
				
				ZseqPassThresoldNot = tf.math.logical_not(ZseqPassThresold)
				ZseqPassThresoldNotInt = tf.dtypes.cast(ZseqPassThresoldNot, tf.int32)
				TseqUpdated = tf.multiply(Tseq[generateParameterNameSeq(l, s, "Tseq")], ZseqPassThresoldNotInt)
				TseqUpdated = tf.add(TseqUpdated, TseqPassThresoldUpdatedValues)
				
				#calculate updated validation matrix
				VseqUpdated = tf.math.logical_or(Vseq[generateParameterNameSeq(l, s, "Vseq")], ZseqPassThresold)
				
				#reset appropriate neurons
				if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
					resetRequiredMatrix = tf.math.logical_and(tf.math.logical_and(ZseqPassThresold, Vseq[generateParameterNameSeq(l, s, "Vseq")]), tf.math.logical_not(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")]))
					for s2 in range(numberOfSequentialInputs):
						Vseq[generateParameterNameSeq(l, s2, "Vseq")] = tf.dtypes.cast(tf.multiply(tf.dtypes.cast(Vseq[generateParameterNameSeq(l, s2, "Vseq")], tf.int32), tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32)), tf.bool)
						if(not doNotResetNeuronOutputUntilAllSequentialInputsActivated):
							Tseq[generateParameterNameSeq(l, s2, "Tseq")] = tf.multiply(Tseq[generateParameterNameSeq(l, s2, "Tseq")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))
							Zseq[generateParameterNameSeq(l, s2, "Zseq")] = tf.multiply(Zseq[generateParameterNameSeq(l, s2, "Zseq")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.float32))
							Aseq[generateParameterNameSeq(l, s2, "Aseq")] = tf.multiply(Aseq[generateParameterNameSeq(l, s2, "Aseq")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.float32))
					sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.dtypes.cast(tf.multiply(tf.dtypes.cast(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")], tf.int32), tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32)), tf.bool)
					if(not doNotResetNeuronOutputUntilAllSequentialInputsActivated):
						Z[generateParameterName(l, "Z")] = tf.multiply(Z[generateParameterName(l, "Z")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.float32))
						A[generateParameterName(l, "A")] = tf.multiply(A[generateParameterName(l, "A")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.float32))
						T[generateParameterName(l, "T")] = tf.multiply(T[generateParameterName(l, "T")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))

				#store updated validation matrix
				Vseq[generateParameterNameSeq(l, s, "Vseq")] = VseqUpdated
				Tseq[generateParameterNameSeq(l, s, "Tseq")] = TseqUpdated
				sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.math.logical_and(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")], Vseq[generateParameterNameSeq(l, s, "Vseq")])
				
				
				#calculate thresholded output
				ZseqThresholded = tf.multiply(ZseqHypothetical, tf.dtypes.cast(ZseqPassThresoldInt, tf.float32))
				
				if(allowMultipleContributingSubinputsPerSequentialInput):
					if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
						ZseqTadjustedThresholded = tf.multiply(ZseqHypotheticalTadjusted, tf.dtypes.cast(ZseqPassThresoldInt, tf.float32))

				#update Zseq
				ZseqUpdated = tf.multiply(Zseq[generateParameterNameSeq(l, s, "Zseq")], tf.dtypes.cast(ZseqPassThresoldNotInt, tf.float32))
				ZseqUpdated = tf.add(ZseqUpdated, ZseqThresholded)
				Zseq[generateParameterNameSeq(l, s, "Zseq")] = ZseqUpdated
				
				if(allowMultipleContributingSubinputsPerSequentialInput):
					if((resetSequentialInputsIfOnlyFirstInputValid) and (s == 0)):
						ZseqTadjustedUpdated = tf.multiply(ZseqTadjusted[generateParameterNameSeq(l, s, "ZseqTadjusted")], tf.dtypes.cast(ZseqPassThresoldNotInt, tf.float32))
						ZseqTadjustedUpdated = tf.add(ZseqTadjustedUpdated, ZseqTadjustedThresholded)
						ZseqTadjusted[generateParameterNameSeq(l, s, "ZseqTadjusted")] = ZseqTadjustedUpdated
				
				
				#regenerate Aseq after Zseq update
				if(allowMultipleSubinputsPerSequentialInput):
					if(performSummationOfSubInputsNonlinear):
						Aseq[generateParameterNameSeq(l, s, "Aseq")] = tf.nn.sigmoid(Zseq[generateParameterNameSeq(l, s, "Zseq")])
					else:
						Aseq[generateParameterNameSeq(l, s, "Aseq")] = Zseq[generateParameterNameSeq(l, s, "Zseq")]
					
				
				#apply weights to input of neuron sequential input
				if(performSummationOfSequentialInputs):
					if(performSummationOfSequentialInputsWeighted):
						multiples = tf.constant([batchSize,1], tf.int32)
						Wtiled = tf.tile(tf.reshape(W[generateParameterName(l, "W")][s], [1, n_h[l]]), multiples)
					
					#these are all used for different methods of sequential input summation
					if(sequentialInputCombinationModeSummation == 1):
						ZseqSum = tf.add(ZseqSum, Zseq[generateParameterNameSeq(l, s, "Zseq")])
					if(sequentialInputCombinationModeSummation == 2):
						AseqSum = tf.math.add(AseqSum, Aseq[generateParameterNameSeq(l, s, "Aseq")])
					if(sequentialInputCombinationModeSummation == 3):
						ZseqWeighted = tf.multiply(Zseq[generateParameterNameSeq(l, s, "Zseq")], Wtiled)
						ZseqWeightedSum = tf.math.add(ZseqWeightedSum, ZseqWeighted)
					if(sequentialInputCombinationModeSummation == 4):	
						AseqWeighted = tf.multiply(Aseq[generateParameterNameSeq(l, s, "Aseq")], Wtiled)
						AseqWeightedSum = tf.math.add(AseqWeightedSum, AseqWeighted)
								
				if(s == numberOfSequentialInputs-1):
					ZseqLast = Zseq[generateParameterNameSeq(l, s, "Zseq")]
					AseqLast = Aseq[generateParameterNameSeq(l, s, "Aseq")]
					VseqLast = Vseq[generateParameterNameSeq(l, s, "Vseq")]
					TseqLast = Tseq[generateParameterNameSeq(l, s, "Tseq")]
								
			if(performSummationOfSequentialInputs):
				if(sequentialInputCombinationModeSummation == 1):
					Z1 = ZseqSum
					if(sequentialInputCombinationModeSummationAveraged):
						Z1 = Z1/numberOfSequentialInputs
					if(performSummationOfSequentialInputsNonlinear):
						A1 = tf.nn.sigmoid(Z1)	#no weights are applied
					else:
						A1 = Z1
				elif(sequentialInputCombinationModeSummation == 2):
					Z1 = AseqSum
					if(sequentialInputCombinationModeSummationAveraged):
						Z1 = Z1/numberOfSequentialInputs
					if(performSummationOfSequentialInputsNonlinear):
						A1 = tf.nn.sigmoid(Z1)	#no weights are applied
					else:
						A1 = Z1	
				elif(sequentialInputCombinationModeSummation == 3):
					Z1 = ZseqWeightedSum
					if(performSummationOfSequentialInputsNonlinear):
						A1 = tf.nn.sigmoid(Z1)
					else:
						A1 = Z1
				elif(sequentialInputCombinationModeSummation == 4):
					Z1 = AseqWeightedSum
					if(performSummationOfSequentialInputsNonlinear):
						A1 = tf.nn.sigmoid(Z1)
					else:
						A1 = Z1
				
				if(performSummationOfSequentialInputsVerify):
					Z1 = tf.multiply(Z1, tf.dtypes.cast(VseqLast, tf.float32))
					A1 = tf.multiply(A1, tf.dtypes.cast(VseqLast, tf.float32))
			else:
				#VseqLastFloat = VseqFloat
				Z1 = ZseqLast
				A1 = AseqLast
						
			A[generateParameterName(l, "A")] = A1
			Z[generateParameterName(l, "Z")] = Z1
			
			T[generateParameterName(l, "T")] = TseqLast
			
			AprevLayer = A[generateParameterName(l, "A")]
			TprevLayer = T[generateParameterName(l, "T")]
	
	#return tf.nn.softmax(Z1)
	return tf.nn.sigmoid(Z1)	#binary classification

	

def defineNeuralNetworkParametersSANI():

	randomNormal = tf.initializers.RandomNormal()

	global n_h_cumulative
	
	if(supportSkipLayers):
		n_h_cumulativeNP = np.zeros((numberOfLayers+1), dtype=int)
		n_h_cumulativeNP[0] = n_h[0]
		#print("n_h_cumulativeNP[0] = ", n_h_cumulativeNP[0])
		
	for l in range(1, numberOfLayers+1):
		#print("\tl = " + str(l))
		for s in range(numberOfSequentialInputs):
			#print("\t\ts = " + str(s))
			if(useSparseTensors):
				if(allowMultipleSubinputsPerSequentialInput):
				
					if(firstSequentialInputHasOnlyOneSubinput and s==0):
						numberSubinputsPerSequentialInputAdjusted = 1
					elif(lastSequentialInputHasOnlyOneSubinput and s==numberOfSequentialInputs-1):
						numberSubinputsPerSequentialInputAdjusted = 1
					else:
						numberSubinputsPerSequentialInputAdjusted = numberSubinputsPerSequentialInput
					
					if(supportSkipLayers):
						#neuronIndex = np.random.randint(0, n_h_cumulativeNP[l-1]+1, n_h[l])
						CseqNP = np.zeros((numberSubinputsPerSequentialInputAdjusted, n_h[l]))
						CseqLayerNP = np.random.randint(0, l, (numberSubinputsPerSequentialInputAdjusted, n_h[l]))	#this can be modified to make local/distant connections more probable
						for i in range(numberSubinputsPerSequentialInputAdjusted):
							for j in range(n_h[l]):
								l2 = CseqLayerNP[i, j]
								CseqNP[i,j] = np.random.randint(0, n_h[l2], 1)
						Cseq[generateParameterNameSeq(l, s, "Cseq")] = tf.Variable(CseqNP, dtype=tf.int32)
						CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")] = tf.Variable(CseqLayerNP, dtype=tf.int32)
					else:
						CseqNP = np.random.randint(0, n_h[l-1]+1, (numberSubinputsPerSequentialInputAdjusted, n_h[l]))	#note +1 is required because np.random.randint generates int between min and max-1
						Cseq[generateParameterNameSeq(l, s, "Cseq")] = tf.Variable(CseqNP, dtype=tf.int32)

					if(performSummationOfSubInputsWeighted):
						WseqNP = np.random.rand(numberSubinputsPerSequentialInputAdjusted, n_h[l])
						Wseq[generateParameterNameSeq(l, s, "Wseq")] = tf.Variable(WseqNP, dtype=tf.float32)
						Bseq[generateParameterNameSeq(l, s, "Bseq")] = tf.Variable(tf.zeros(n_h[l]), dtype=tf.float32)
				else:
					if(supportSkipLayers):
						#neuronIndex = np.random.randint(0, n_h_cumulativeNP[l-1]+1, n_h[l])
						CseqNP = np.zeros((n_h[l]))
						CseqLayerNP = np.random.randint(0, l, n_h[l])	#this can be modified to make local/distant connections more probable
						for index, l2 in enumerate(CseqLayerNP):
							CseqNP[index] = np.random.randint(0, n_h[l2], 1)
						Cseq[generateParameterNameSeq(l, s, "Cseq")] = tf.Variable(CseqNP, dtype=tf.int32)
						CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")] = tf.Variable(CseqLayerNP, dtype=tf.int32)
					else:
						CseqNP = np.random.randint(0, n_h[l-1]+1, n_h[l])	#note +1 is required because np.random.randint generates int between min and max-1
						Cseq[generateParameterNameSeq(l, s, "Cseq")] = tf.Variable(CseqNP, dtype=tf.int32)
			else:
				if(supportSkipLayers):
					Wseq[generateParameterNameSeq(l, s, "Wseq")] = tf.Variable(randomNormal([n_h_cumulativeNP[l-1], n_h[l]], dtype=tf.float32))
					Bseq[generateParameterNameSeq(l, s, "Bseq")] = tf.Variable(tf.zeros(n_h[l]), dtype=tf.float32)
				else:
					Wseq[generateParameterNameSeq(l, s, "Wseq")] = tf.Variable(randomNormal([n_h[l-1], n_h[l]], dtype=tf.float32))
					Bseq[generateParameterNameSeq(l, s, "Bseq")] = tf.Variable(tf.zeros(n_h[l]), dtype=tf.float32)

		if(performSummationOfSequentialInputsWeighted):	
			W[generateParameterName(l, "W")] = tf.Variable(randomNormal([numberOfSequentialInputs, n_h[l]], dtype=tf.float32))	#randomNormal	#note the architecture of this weight matrix is different than a normal weight matrix. Every set of numberOfSequentialInputs (e.g. 3) represents a unique set of sequential neuron inputs (artificial neurons), and are mapped to an independent neuron (real neuron)
			B[generateParameterName(l, "B")] = tf.Variable(tf.zeros(n_h[l]), tf.float32)
					
		if(supportSkipLayers):
			n_h_cumulativeNP[l] = n_h_cumulativeNP[l-1] + n_h[l]
			  
	if(supportSkipLayers):
		n_h_cumulative['n_h_cumulative'] = tf.Variable(n_h_cumulativeNP, dtype=tf.int32)

def printShape(tensor, tensorName):
	print(tensorName + ".shape = ")
	print(tensor.shape)
	
