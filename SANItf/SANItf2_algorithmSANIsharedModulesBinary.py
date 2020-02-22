# -*- coding: utf-8 -*-
"""SANItf2_algorithmSANIsharedModulesBinary.py

# Requirements:
Python 3 and Tensorflow 2.1+ 

# License:
MIT License

# Execution:
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

numberOfFeaturesPerWord = -1
paddingTagIndex = -1

sequentialInputActivationThreshold = 0.1	#CHECKTHIS


allowMultipleSubinputsPerSequentialInput = False

numberOfWordsInConvolutionalWindowSeen = 1	#CHECKTHIS
	
resetSequentialInputsIfOnlyFirstInputValid = True	#see GIA_TXT_REL_TRANSLATOR_NEURAL_NETWORK_SEQUENCE_GRAMMAR development history for meaning and algorithmic implications of this feature
if(resetSequentialInputsIfOnlyFirstInputValid):
	doNotResetNeuronOutputUntilAllSequentialInputsActivated = True

useSparseTensors = True	#mandatory

#if(numberOfSequentialInputs == 2):
oneSequentialInputHasOnlyOneSubinput = True	#conditional probability determination of events
if(oneSequentialInputHasOnlyOneSubinput):
	firstSequentialInputHasOnlyOneSubinput = True #use combination of allowMultipleSubinputsPerSequentialInput for different sequential inputs;  #1[#2] sequential input should allow multiple subinputs, #2[#1] sequential input should allow single subinput
	if(firstSequentialInputHasOnlyOneSubinput):
		lastSequentialInputHasOnlyOneSubinput = False
	else:
		lastSequentialInputHasOnlyOneSubinput = True

supportSkipLayers = True
if(oneSequentialInputHasOnlyOneSubinput):
	numberSubinputsPerSequentialInput = 50	#~approx equal number of prev layer neurons/2	#number of prior/future events in which to calculate a conditional probability
else:
	numberSubinputsPerSequentialInput = 3	#sparsity
	
useLastSequentialInputOnly = True	#implied variable (not used)

		
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

	batchSize = 50
	displayStep = 100

	return learningRate, trainingSteps, batchSize, displayStep
	
def defineNetworkParametersSANI(num_input_neurons, num_output_neurons, datasetNumFeatures, dataset, trainMultipleFiles):

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
		
		n_h_1 = int(datasetNumFeatures*1)	#*x for skip layers	
		n_h_2 = int(datasetNumFeatures*2)
		n_h_3 = int(datasetNumFeatures*3)
		n_h_4 = int(datasetNumFeatures*4)
		n_h_5 = n_y
		n_h = [n_h_0, n_h_1, n_h_2, n_h_3, n_h_4, n_h_5]

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
	
	
def neuralNetworkPropagationSANI(x):
		
	#note connectivity indexes are used rather than sparse weight matrices due to limitations in current tf2 sparse tensor implementation
	
	#definitions for reference:
	
	#neuron sequential input vars;
	#x/AprevLayer	#output vector (dim: batchSize*n_h[l])

	#Cseq	#static connectivity matrix (int) - indexes of neurons on prior layer stored; mapped to W  (dim: numberSubinputsPerSequentialInput*n_h[l])
	#if(supportSkipLayers):
		#CseqLayer	#static connectivity matrix (int) - indexes of pior layer stored; mapped to W (dim: numberSubinputsPerSequentialInput*n_h[l])
	#AseqSum	#combination variable

	#Vseq	#mutable verification vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)			#records whether particular neuron sequential inputs are currently active
	#Tseq	#mutable time vector (dim: batchSize*n_h[l] - regenerated for each sequential input index)				#records the time at which a particular sequential input last fired
	#Zseq	#neuron activation function output vector (dim: batchSize*n_h[l]  - regenerated for each sequential input index)	#records the current output value of sequential inputs
	#Aseq	#neuron activation function output vector (dim: batchSize*n_h[l]  - regenerated for each sequential input index)	#records the current output value of sequential inputs
	
	#Q  
	#Z	#neuron activation function input (dim: batchSize*n_h[l])
	#A	#neuron activation function output (dim: batchSize*n_h[l])
	#T	#mutable time vector (dim: batchSize*n_h[l]) - same as Tseq[numberOfSequentialInputs-1]
	#if(performSummationOfSequentialInputsWeighted):	
		#W	(dim: numberOfSequentialInputs*n_h[l])	
		
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
		Z[generateParameterName(l, "Z")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), dtype=tf.bool), dtype=tf.bool)	#A=Z for binary activations
		A[generateParameterName(l, "A")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), dtype=tf.bool), dtype=tf.bool)	#A=Z for binary activations
		T[generateParameterName(l, "T")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.int32))
		for s in range(numberOfSequentialInputs):
			Vseq[generateParameterNameSeq(l, s, "Vseq")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), tf.bool), dtype=tf.bool)
			Tseq[generateParameterNameSeq(l, s, "Tseq")] = tf.Variable(tf.zeros([batchSize, n_h[l]], dtype=tf.int32))
			Zseq[generateParameterNameSeq(l, s, "Zseq")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), dtype=tf.bool), dtype=tf.bool)	#A=Z for binary activations
			Aseq[generateParameterNameSeq(l, s, "Aseq")] = tf.Variable(tf.dtypes.cast(tf.zeros([batchSize, n_h[l]]), dtype=tf.bool), dtype=tf.bool)	#A=Z for binary activations

	#for w in range(maxNumberOfWordsInSentenceBatch):
	#for w in range(0, 1):
	for w in range(maxNumberOfWordsInSentenceBatch-numberOfWordsInConvolutionalWindowSeen+1):
	
		print("w = " + str(w))
		
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
			
		AprevLayer = AfirstLayerShifted
		
		TprevLayer = tf.ones(([batchSize, n_h[0]]), tf.int32)
		TprevLayer = TprevLayer*w
		
		for l in range(1, numberOfLayers+1):	#start algorithm at n_h[1]; ie first hidden layer

			print("\tl = " + str(l))

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

			for sForward in range(numberOfSequentialInputs):
				sReverse = numberOfSequentialInputs-sForward-1	
				s = sReverse
				
				#for each sequential input of each neuron (stating from last), see if its requirements are satisfied
					#if first sequential input, and hypothetical valid activation of first input, allow reset of neuron sequential input 
						#ie if((s == 0) && (resetSequentialInputsIfOnlyFirstInputValid)):
							#skip sequential validation requirements as neuron sequential input can be reset
				
				print("\t\ts = " + str(s))

				#identify (hypothetical) activation of neuron sequential input
				if(supportSkipLayers):
					CseqCrossLayer = tf.add(Cseq[generateParameterNameSeq(l, s, "Cseq")], tf.gather(n_h_cumulative['n_h_cumulative'], CseqLayer[generateParameterNameSeq(l, s, "CseqLayer")]))					
					AseqInput = tf.gather(AprevLayerAll, CseqCrossLayer, axis=1)
					TseqInput = tf.gather(TprevLayerAll, CseqCrossLayer, axis=1)
				else:
					AseqInput = tf.gather(AprevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)
					TseqInput = tf.gather(TprevLayer, Cseq[generateParameterNameSeq(l, s, "Cseq")], axis=1)

				
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
				
				#AseqInput = tf.multiply(VseqBool, AseqInput)
			
				#calculate output for layer sequential input s

				#ensure that T continguous constraint is met;
				#NO: take the max subinput pathway only (ie matrix mult but with max() rather than sum() for each dot product)
				if(s > 0):
					#1. first select the AseqInput inputs that have T[l-1] contiguous with Tseq[l, s-1]:
					TseqPlus1 = Tseq[generateParameterNameSeq(l, s-1, "Tseq")]+1

					multiples = tf.constant([1,numberSubinputsPerSequentialInput,1], tf.int32)

					TseqPlus1Tiled = tf.tile(tf.reshape(TseqPlus1, [batchSize, 1, n_h[l]]), multiples)
					TseqInputThreshold = tf.math.equal(TseqInput, TseqPlus1Tiled)

					AseqInputTthresholded = tf.math.logical_and(AseqInput, TseqInputThreshold)
					AseqInput = AseqInputTthresholded
				else:
					if(resetSequentialInputsIfOnlyFirstInputValid):
						#only reset first sequential input if TseqInput > T[l]

						Ttiled = tf.tile(tf.reshape(T[generateParameterName(l, "T")], [batchSize, 1, n_h[l]]), multiples)
						VseqTiled = tf.tile(tf.reshape(Vseq[generateParameterNameSeq(l, s, "Vseq")], [batchSize, 1, n_h[l]]), multiples)

						ZseqHypotheticalResetThreshold = tf.math.logical_or(tf.math.logical_not(VseqTiled), tf.math.greater(TseqInput, tf.dtypes.cast(Ttiled, tf.int32)))

						AseqInputTthresholded = tf.math.logical_and(AseqInput, ZseqHypotheticalResetThreshold)
						AseqInput = AseqInputTthresholded


				#take any active and Tthreshold valid sub input:
				ZseqHypothetical = tf.math.reduce_any(AseqInput, axis=1)
					
				#apply sequential validation matrix
				ZseqHypothetical = tf.math.logical_and(VseqBool, ZseqHypothetical)
				
				#check output threshold
				ZseqPassThresold = ZseqHypothetical	#redundant
				
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
							Zseq[generateParameterNameSeq(l, s2, "Zseq")] = tf.math.logical_and(Zseq[generateParameterNameSeq(l, s2, "Zseq")], tf.math.logical_not(resetRequiredMatrix))
							Aseq[generateParameterNameSeq(l, s2, "Aseq")] = tf.math.logical_and(Aseq[generateParameterNameSeq(l, s2, "Aseq")], tf.math.logical_not(resetRequiredMatrix))
					sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.dtypes.cast(tf.multiply(tf.dtypes.cast(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")], tf.int32), tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32)), tf.bool)
					if(not doNotResetNeuronOutputUntilAllSequentialInputsActivated):
						Z[generateParameterName(l, "Z")] = tf.math.logical_and(Z[generateParameterName(l, "Z")], tf.math.logical_not(resetRequiredMatrix))
						A[generateParameterName(l, "A")] = tf.math.logical_and(A[generateParameterName(l, "A")], tf.math.logical_not(resetRequiredMatrix))
						T[generateParameterName(l, "T")] = tf.multiply(T[generateParameterName(l, "T")], tf.dtypes.cast(tf.math.logical_not(resetRequiredMatrix), tf.int32))

				#store updated validation matrix
				Vseq[generateParameterNameSeq(l, s, "Vseq")] = VseqUpdated
				Tseq[generateParameterNameSeq(l, s, "Tseq")] = TseqUpdated
				sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")] = tf.math.logical_and(sequentialActivationFound[generateParameterName(l, "sequentialActivationFound")], Vseq[generateParameterNameSeq(l, s, "Vseq")])
				
				#calculate thresholded output
				ZseqThresholded = ZseqPassThresold	#redundant

				#update Zseq
				Zseq[generateParameterNameSeq(l, s, "Zseq")] = tf.math.logical_or(Zseq[generateParameterNameSeq(l, s, "Zseq")], ZseqPassThresold) 
			
				#regenerate Aseq after Zseq update
				Aseq[generateParameterNameSeq(l, s, "Aseq")] = Zseq[generateParameterNameSeq(l, s, "Zseq")]
								
				if(s == numberOfSequentialInputs-1):
					ZseqLast = Zseq[generateParameterNameSeq(l, s, "Zseq")]
					AseqLast = Aseq[generateParameterNameSeq(l, s, "Aseq")]
					VseqLast = Vseq[generateParameterNameSeq(l, s, "Vseq")]
					TseqLast = Tseq[generateParameterNameSeq(l, s, "Tseq")]
								
			Z1 = ZseqLast
			A1 = AseqLast
						
			A[generateParameterName(l, "A")] = A1
			Z[generateParameterName(l, "Z")] = Z1
			
			T[generateParameterName(l, "T")] = TseqLast
			
			AprevLayer = A[generateParameterName(l, "A")]
			TprevLayer = T[generateParameterName(l, "T")]
	
	#return tf.nn.softmax(Z1)
	#return tf.nn.sigmoid(Z1)	#binary classification
	return Z1

	

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
			
			if(firstSequentialInputHasOnlyOneSubinput and s==0):
				numberSubinputsPerSequentialInputAdjusted = 1
			elif(lastSequentialInputHasOnlyOneSubinput and s==numberOfSequentialInputs-1):
				numberSubinputsPerSequentialInputAdjusted = 1
			else:
				numberSubinputsPerSequentialInputAdjusted = numberSubinputsPerSequentialInput

			if(supportSkipLayers):
				#neuronIndex = np.random.randint(0, n_h_cumulativeNP[l-1]+1, n_h[l])
				CseqNP = np.zeros((numberSubinputsPerSequentialInput, n_h[l]))
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
		
		if(supportSkipLayers):
			n_h_cumulativeNP[l] = n_h_cumulativeNP[l-1] + n_h[l]
			  
	if(supportSkipLayers):
		n_h_cumulative['n_h_cumulative'] = tf.Variable(n_h_cumulativeNP, dtype=tf.int32)

def printShape(tensor, tensorName):
	print(tensorName + ".shape = ")
	print(tensor.shape)
	
